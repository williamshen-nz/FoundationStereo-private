# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import rerun as rr
import os, sys

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import logging

try:
    import pyzed.sl as sl
except ImportError:
    print("Error: ZED SDK not installed. Please install pyzed:")
    print("pip install pyzed")
    import sys

    sys.exit(1)


def init_zed_camera(resolution="HD720", fps=30, serial_number=None):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
    init_params.camera_fps = fps
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER

    if serial_number:
        init_params.set_from_serial_number(serial_number)

    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")

    camera_info = zed.get_camera_information()
    logging.info(
        f"ZED camera SN {camera_info.serial_number}: "
        f"{camera_info.camera_configuration.resolution.width}x"
        f"{camera_info.camera_configuration.resolution.height}"
    )

    return zed, sl.RuntimeParameters()


def capture_zed_stereo(zed, runtime_params, resolution):
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        left_image = sl.Mat()
        right_image = sl.Mat()

        zed.retrieve_image(left_image, sl.VIEW.LEFT, sl.MEM.CPU, resolution)
        zed.retrieve_image(right_image, sl.VIEW.RIGHT, sl.MEM.CPU, resolution)

        left_bgra = left_image.get_data()
        right_bgra = right_image.get_data()

        # Drop alpha channel, keep BGR order
        left_bgr = left_bgra[:, :, :3].copy()
        right_bgr = right_bgra[:, :, :3].copy()

        return left_bgr, right_bgr
    else:
        return None, None


def get_zed_calibration(zed, scale=1.0):
    camera_info = zed.get_camera_information()
    calibration_params = camera_info.camera_configuration.calibration_parameters
    left_cam = calibration_params.left_cam

    K = np.array(
        [
            [left_cam.fx * scale, 0, left_cam.cx * scale],
            [0, left_cam.fy * scale, left_cam.cy * scale],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    baseline = calibration_params.get_camera_baseline()
    logging.info(f"ZED calibration: fx={left_cam.fx:.2f}, baseline={baseline:.4f}m")

    return K, baseline


def load_model():
    cfg = OmegaConf.load(f"{os.path.dirname(args.ckpt_dir)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    for k, v in args.__dict__.items():
        cfg[k] = v
    model_args = OmegaConf.create(cfg)

    model = FoundationStereo(model_args)
    ckpt = torch.load(args.ckpt_dir)
    logging.info(
        f"Loading checkpoint: global_step={ckpt['global_step']}, epoch={ckpt['epoch']}"
    )
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        default=f"{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth",
        type=str,
        help="pretrained model path",
    )
    parser.add_argument(
        "--out_dir",
        default=f"{code_dir}/../output_zed/",
        type=str,
        help="the directory to save results",
    )
    parser.add_argument(
        "--scale",
        default=1,
        type=float,
        help="downsize the image by scale, must be <=1",
    )
    parser.add_argument(
        "--hiera",
        default=0,
        type=int,
        help="hierarchical inference (only needed for high-resolution images (>1K))",
    )
    parser.add_argument(
        "--z_far", default=10, type=float, help="max depth to clip in point cloud"
    )
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=32,
        help="number of flow-field updates during forward pass",
    )
    parser.add_argument("--get_pc", type=int, default=1, help="save point cloud output")
    parser.add_argument(
        "--remove_invisible",
        default=1,
        type=int,
        help="remove non-overlapping observations between left and right images from point cloud",
    )
    args = parser.parse_args()
    rr.init("zed_demo", spawn=True)

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize ZED camera
    zed, runtime_params = init_zed_camera(
        resolution="HD720", fps=30, serial_number=16779706
    )
    camera_info = zed.get_camera_information()
    resolution = camera_info.camera_configuration.resolution
    left_image, right_image = capture_zed_stereo(zed, runtime_params, resolution)

    # Convert BGR to RGB
    img0 = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
    rr.log("img0", rr.Image(img0))
    rr.log("img1", rr.Image(img1))
    imageio.imwrite(f"{args.out_dir}/left_captured.png", img0)
    imageio.imwrite(f"{args.out_dir}/right_captured.png", img1)

    # Get calibration
    K, baseline = get_zed_calibration(zed, scale=1.0)
    zed.close()

    # Preprocess images
    if args.scale < 1:
        img0 = cv2.resize(img0, fx=args.scale, fy=args.scale, dsize=None)
        img1 = cv2.resize(img1, fx=args.scale, fy=args.scale, dsize=None)
        K = K.copy()
        K[:2] *= args.scale

    H, W = img0.shape[:2]
    img0_ori = img0.copy()

    img0_tensor = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1_tensor = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_tensor.shape, divis_by=32, force_square=False)
    img0_tensor, img1_tensor = padder.pad(img0_tensor, img1_tensor)

    # Run stereo inference
    model = load_model()
    start_time = time.time()
    with torch.cuda.amp.autocast(True):
        if not args.hiera:
            disp = model.forward(
                img0_tensor, img1_tensor, iters=args.valid_iters, test_mode=True
            )
        else:
            disp = model.run_hierachical(
                img0_tensor,
                img1_tensor,
                iters=args.valid_iters,
                test_mode=True,
                small_ratio=0.5,
            )

    torch.cuda.synchronize()
    inference_time = time.time() - start_time
    logging.info(f"Inference: {inference_time:.3f}s ({1.0 / inference_time:.2f} FPS)")

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)

    valid_disp = disp[np.isfinite(disp)]
    if len(valid_disp) > 0:
        logging.info(
            f"Disparity: min={valid_disp.min():.2f}, max={valid_disp.max():.2f}, "
            f"mean={valid_disp.mean():.2f}"
        )

    # Save disparity visualization
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    imageio.imwrite(f"{args.out_dir}/vis.png", vis)

    # Generate depth and point cloud
    if args.get_pc:
        # Convert disparity to depth
        if args.remove_invisible:
            yy, xx = np.meshgrid(
                np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij"
            )
            us_right = xx - disp
            invalid = us_right < 0
            disp = disp.copy()
            disp[invalid] = np.inf

        depth = K[0, 0] * baseline / disp
        valid_depth = depth[np.isfinite(depth)]
        if len(valid_depth) > 0:
            logging.info(
                f"Depth: min={valid_depth.min():.2f}m, max={valid_depth.max():.2f}m"
            )
        np.save(f"{args.out_dir}/depth_meter.npy", depth)
        rr.log("depth", rr.DepthImage(depth, meter=1.0))

        # Generate point cloud
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))

        # keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (
        #     np.asarray(pcd.points)[:, 2] <= args.z_far
        # )
        # keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        # pcd = pcd.select_by_index(keep_ids)
        # logging.info(f"Point cloud: {len(pcd.points)} points (0 < z <= {args.z_far}m)")

        o3d.io.write_point_cloud(f"{args.out_dir}/cloud.ply", pcd)

        # Visualize
        # logging.info("Visualizing point cloud. Press ESC to exit.")
        # vis_3d = o3d.visualization.Visualizer()
        # vis_3d.create_window()
        # vis_3d.add_geometry(pcd)
        # vis_3d.get_render_option().point_size = 1.0
        # vis_3d.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
        # vis_3d.run()
        # vis_3d.destroy_window()
        rr.log("pcd", rr.Points3D(positions=pcd.points, colors=pcd.colors))

