import numpy as np
import json
import os
from tqdm import tqdm

from common import *
from scenes import scenes_nerf, scenes_image, scenes_sdf, scenes_volume, setup_colored_sdf
import pyngp as ngp # noqa
from os.path import join

def load_ref_images(args, transform_path):
    with open(transform_path) as f:
        test_transforms = json.load(f)
    if os.path.isfile(args.scene):
        data_dir=os.path.dirname(args.scene)
    else:
        data_dir=args.scene
    
    n_camera_views = len(test_transforms["frames"])
    ref_images = []
    for camera_view in tqdm(range(n_camera_views)):
        frame = test_transforms["frames"][camera_view]
        p = frame["file_path"]
        if "." not in p:
            p = p + ".png"
        ref_fname = os.path.join(data_dir, p)
        if not os.path.isfile(ref_fname):
            ref_fname = os.path.join(data_dir, p + ".png")
            if not os.path.isfile(ref_fname):
                ref_fname = os.path.join(data_dir, p + ".jpg")
                if not os.path.isfile(ref_fname):
                    ref_fname = os.path.join(data_dir, p + ".jpeg")
                    if not os.path.isfile(ref_fname):
                        ref_fname = os.path.join(data_dir, p + ".exr")
        ref_image = read_image(ref_fname)
        ref_images.append(ref_image)
    return ref_images

def load_obj_mesh(mesh_file, with_normal=False, with_texture=False):
    vertex_data = []
    norm_data = []
    uv_data = []


    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1

    if with_texture and with_normal:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        norms = np.array(norm_data)
        if norms.shape[0] == 0:
            norms = compute_normal(vertices, faces)
            face_normals = faces
        else:
            norms = normalize_v3(norms)
            face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals, uvs, face_uvs

    if with_texture:
        uvs = np.array(uv_data)
        face_uvs = np.array(face_uv_data) - 1
        return vertices, faces, uvs, face_uvs

    if with_normal:
        norms = np.array(norm_data)
        norms = normalize_v3(norms)
        face_normals = np.array(face_norm_data) - 1
        return vertices, faces, norms, face_normals

    return vertices, faces

def cal_psnr(testbed, ref_images):
    print("calculate psnr...")
    psnr_l = []
    
    testbed.background_color = [0.0, 0.0, 0.0, 0.0]
    
    # Prior nerf papers don't typically do multi-sample anti aliasing.
    # So snap all pixels to the pixel centers.
    testbed.snap_to_pixel_centers = True
    spp = 8

    testbed.nerf.rendering_min_transmittance = 1e-4
    
    for camera_view, ref_image in tqdm(enumerate(ref_images)):
        if camera_view % 5 != 0:
            continue
        if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
            # Since sRGB conversion is non-linear, alpha must be factored out of it
            ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
            ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
            ref_image[...,:3] *= ref_image[...,3:4]
            ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
            ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])
        testbed.set_camera_to_training_view(camera_view)
        image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)
        A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
        R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
        mse = float(compute_error("MSE", A, R))
        psnr = mse2psnr(mse)
        psnr_l.append(psnr)
    psnr_l = np.array(psnr_l)
    avg_psnr = np.average(psnr_l)
    return avg_psnr
        

def render_img_training_view(args, testbed, log_ptr, image_dir, frame_time_id = 0, training_step = -1):
    eval_path = args.output_path
    os.makedirs(eval_path, exist_ok=True)
    img_path = os.path.join(eval_path, 'images',f'{args.test_camera_view:04}')
    os.makedirs(img_path, exist_ok=True)
    # log_path = os.path.join(eval_path, f"eval_log.txt")
    # log_ptr = open(log_path, "w+")
    print("Evaluating test transforms from ", args.scene, file=log_ptr)
    log_ptr.flush()
    with open(image_dir) as f:
        test_transforms = json.load(f)
    if os.path.isfile(args.scene):
        data_dir=os.path.dirname(args.scene)
    else:
        data_dir=args.scene
    totmse = 0
    totpsnr = 0
    totssim = 0
    totcount = 0
    minpsnr = 1000
    maxpsnr = 0

    # Evaluate metrics on black background
    # testbed.background_color = [0.0, 0.0, 0.0, 1.0]
    testbed.background_color = [0.0, 0.0, 0.0, 0.0]
    # testbed.background_color = [1.0, 1.0, 1.0, 0.0]

    # Prior nerf papers don't typically do multi-sample anti aliasing.
    # So snap all pixels to the pixel centers.
    testbed.snap_to_pixel_centers = True
    spp = 8

    testbed.nerf.rendering_min_transmittance = 1e-4

    # assert "from_na" in test_transforms.keys(), print("only for na_data")

    camera_view = args.test_camera_view
    frame = test_transforms["frames"][camera_view]
    p = frame["file_path"]
    if "." not in p:
        p = p + ".png"
    ref_fname = os.path.join(data_dir, p)
    if not os.path.isfile(ref_fname):
        ref_fname = os.path.join(data_dir, p + ".png")
        if not os.path.isfile(ref_fname):
            ref_fname = os.path.join(data_dir, p + ".jpg")
            if not os.path.isfile(ref_fname):
                ref_fname = os.path.join(data_dir, p + ".jpeg")
                if not os.path.isfile(ref_fname):
                    ref_fname = os.path.join(data_dir, p + ".exr")

    ref_image = read_image(ref_fname)

            # NeRF blends with background colors in sRGB space, rather than first
            # transforming to linear space, blending there, and then converting back.
            # (See e.g. the PNG spec for more information on how the `alpha` channel
            # is always a linear quantity.)
            # The following lines of code reproduce NeRF's behavior (if enabled in
            # testbed) in order to make the numbers comparable.
    if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
        # Since sRGB conversion is non-linear, alpha must be factored out of it
        ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
        ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
        ref_image[...,:3] *= ref_image[...,3:4]
        ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
        ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])
    
    if training_step < 0:
        write_image(join(img_path,f"frame_{frame_time_id:06}_gt.png"), ref_image)
    else:
        os.makedirs(os.path.join(img_path, f"frame_{frame_time_id:06}"), exist_ok=True)
        write_image(join(img_path,f"frame_{frame_time_id:06}",f"frame_{frame_time_id:06}_{training_step}_gt.png"), ref_image)

    testbed.set_camera_to_training_view(camera_view)
    image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

    if training_step < 0:
        write_image(join(img_path,f"frame_{frame_time_id:06}_pred.png"), image)
    else:
        write_image(join(img_path,f"frame_{frame_time_id:06}",f"frame_{frame_time_id:06}_{training_step}_pred.png"), image)
    
    diffimg = np.absolute(image - ref_image)
    diffimg[...,3:4] = 1.0

    if training_step < 0:
        write_image(join(img_path,f"frame_{frame_time_id:06}_diff.png"), diffimg)
    else:
        write_image(join(img_path,f"frame_{frame_time_id:06}",f"frame_{frame_time_id:06}_{training_step}_diff.png"), diffimg)
        return
    
    A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
    R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
    mse = float(compute_error("MSE", A, R))
    ssim = float(compute_error("SSIM", A, R))
    totssim += ssim
    totmse += mse
    psnr = mse2psnr(mse)
    totpsnr += psnr
    minpsnr = psnr if psnr<minpsnr else minpsnr
    maxpsnr = psnr if psnr>maxpsnr else maxpsnr
    totcount = totcount+1

    psnr_avgmse = mse2psnr(totmse/(totcount or 1))
    psnr = totpsnr/(totcount or 1)
    ssim = totssim/(totcount or 1)
    print(f"camera_view:{camera_view}, frame_time:{frame_time_id}: PSNR={psnr} SSIM={ssim}", file=log_ptr)
    log_ptr.flush() # write immediately to file

    ## render mesh normal
    mesh = trimesh.load(args.save_mesh)
    vex = mesh.vertices
    faces = mesh.faces
    # vex, faces = load_obj_mesh(args.save_mesh)
    # with open(all_transform_path[testbed.current_training_time_frame]) as f:
        # this_frame_transforms = json.load(f)
    # frame = this_frame_transforms['frames'][camera_view]
    ext = torch.inverse(torch.tensor(np.array(frame['transform_matrix'])))
    ixt = (torch.tensor(np.array(frame['intrinsic_matrix'])))

    # scale_ratio = torch.tensor([1024/1285,1024/940]) # Franzi
    # scale_ratio = torch.tensor([1024/512,1024/512]) # Franzi
    # scale_ratio = torch.tensor([ref_image.shape[1]/512,ref_image.shape[0]/512]) # Franzi
    scale_ratio = torch.tensor([1024/ref_image.shape[1], 1024/ref_image.shape[0]]) # Franzi

    if args.scene[-4:] == "json":
        base_dir = os.path.dirname(args.scene) 
    else:
        base_dir = args.scene

    # apply smpl's rotation and transition
    # smpl_transform_dir = join(base_dir, test_transforms['smpl']['transform_dir'])
    # with open(smpl_transform_dir) as f:
    #     smpl_transforms = json.load(f)

    # Rh = smpl_transforms['Rh']
    # Th = torch.tensor(smpl_transforms['Th']).float()
    # Rh = np.array(Rh) # world_to_local
    # R = torch.tensor(cv2.Rodrigues(Rh)[0]).float()
  
    # ext = torch.tensor(frame['transform_matrix'])
    # ext[:3,:3] = torch.matmul(R.T, ext[:3,:3])
    # ext[:3,3] = torch.matmul(R.T, ext[:3,3] - Th[0]/0.4)
    # ext = torch.inverse(ext)
    

    # mesh.vertices = torch.matmul(torch.tensor(mesh.vertices).float(), R.T) + Th[0]
    # mesh.vertices /= 0.4
    ixt[:2,:] = ixt[:2,:] * scale_ratio[:,None]

    normal_img = render_mesh(renderer, vex.astype(np.float32), faces, ixt, ext)
    cv2.imwrite(join(img_path,f"frame_{frame_time_id:06}_mesh.png"),normal_img)
    # write_image(join(img_path,f"frame_{frame_time_id:06}_mesh.png"), normal_img) ## have bugs, all white


    # Use white background to show the results
    testbed.background_color = [1.0, 1.0, 1.0, 1.0]

    # Prior nerf papers don't typically do multi-sample anti aliasing.
    # So snap all pixels to the pixel centers.
    testbed.snap_to_pixel_centers = True
    spp = 8

    testbed.nerf.rendering_min_transmittance = 1e-4

    # assert "from_na" in test_transforms.keys(), print("only for na_data")

    camera_view = args.test_camera_view
    frame = test_transforms["frames"][camera_view]
    p = frame["file_path"]
    if "." not in p:
        p = p + ".png"
    ref_fname = os.path.join(data_dir, p)
    if not os.path.isfile(ref_fname):
        ref_fname = os.path.join(data_dir, p + ".png")
        if not os.path.isfile(ref_fname):
            ref_fname = os.path.join(data_dir, p + ".jpg")
            if not os.path.isfile(ref_fname):
                ref_fname = os.path.join(data_dir, p + ".jpeg")
                if not os.path.isfile(ref_fname):
                    ref_fname = os.path.join(data_dir, p + ".exr")

    ref_image = read_image(ref_fname)

    # NeRF blends with background colors in sRGB space, rather than first
    # transforming to linear space, blending there, and then converting back.
    # (See e.g. the PNG spec for more information on how the `alpha` channel
    # is always a linear quantity.)
    # The following lines of code reproduce NeRF's behavior (if enabled in
    # testbed) in order to make the numbers comparable.
    if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
        # Since sRGB conversion is non-linear, alpha must be factored out of it
        ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
        ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
        ref_image[...,:3] *= ref_image[...,3:4]
        ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
        ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])
    elif ref_image.shape[2] == 4:
        ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
        
    testbed.set_camera_to_training_view(camera_view)
    image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

    return load_image(image), load_image(ref_image), normal_img

# import platform
# if not platform.system().lower() == 'windows':
from pytorch3d_utils import *
