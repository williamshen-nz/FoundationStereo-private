# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import rerun as rr
import os, sys
import argparse

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{code_dir}/../")

from Utils import *
from stereo_client import StereoClient

try:
    import pyzed.sl as sl
except ImportError:
    print("Error: ZED SDK not installed. Please install pyzed:")
    print("pip install pyzed")
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='http://localhost:1234',
                        help='Server URL')
    parser.add_argument('--out_dir', default=f'{code_dir}/../output_zed_client/',
                        type=str, help='Output directory')
    parser.add_argument('--scale', default=1.0, type=float,
                        help='Image downscale factor')
    parser.add_argument('--hiera', default=0, type=int,
                        help='Use hierarchical inference')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='Number of iterations')
    parser.add_argument('--z_far', default=10, type=float,
                        help='Max depth for point cloud')
    parser.add_argument('--format', choices=['npz', 'png'], default='npz',
                        help='Response format from server')
    parser.add_argument('--get_pc', type=int, default=1,
                        help='Generate point cloud')
    args = parser.parse_args()

    rr.init("zed_client_demo", spawn=True)
    set_logging_format()
    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize client and check server health
    client = StereoClient(args.server)
    if not client.health_check():
        logging.error(f"Server at {args.server} is not healthy!")
        logging.error("Make sure server is running: python scripts/server.py")
        sys.exit(1)
    logging.info(f"Server is healthy at {args.server}")

    # Initialize ZED camera
    zed, runtime_params = init_zed_camera(
        resolution='HD720',
        fps=30,
        serial_number=16779706
    )

    # Capture stereo pair
    camera_info = zed.get_camera_information()
    resolution = camera_info.camera_configuration.resolution
    left_image, right_image = capture_zed_stereo(zed, runtime_params, resolution)

    # Convert BGR to RGB
    img0 = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

    rr.log("img0", rr.Image(img0))
    rr.log("img1", rr.Image(img1))

    # Get calibration
    K, baseline = get_zed_calibration(zed, scale=1.0)
    zed.close()

    # Save captured images
    imageio.imwrite(f'{args.out_dir}/left_captured.png', img0)
    imageio.imwrite(f'{args.out_dir}/right_captured.png', img1)
    logging.info(f"Captured images saved to {args.out_dir}")

    # Send to server for inference
    logging.info(f"Sending to server {args.server} (format: {args.format})...")
    depth, inference_time = client.infer_from_arrays(
        img0, img1, K, baseline,
        scale=args.scale,
        hiera=bool(args.hiera),
        valid_iters=args.valid_iters,
        format=args.format
    )
    logging.info(f"Server inference time: {inference_time:.3f}s")

    # Log depth stats
    valid_depth = depth[np.isfinite(depth)]
    if len(valid_depth) > 0:
        logging.info(f"Depth: min={valid_depth.min():.2f}m, max={valid_depth.max():.2f}m, "
                     f"mean={valid_depth.mean():.2f}m")

    # Save depth
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    rr.log("depth", rr.DepthImage(depth, meter=1.0))
    logging.info(f"Depth saved to {args.out_dir}/depth_meter.npy")

    # Visualize disparity (convert depth back for visualization)
    with np.errstate(divide='ignore', invalid='ignore'):
        disp = K[0, 0] * baseline / depth
    disp[~np.isfinite(disp)] = 0
    vis = vis_disparity(disp)
    vis = np.concatenate([img0, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/vis.png', vis)

    # Generate point cloud
    if args.get_pc:
        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0.reshape(-1, 3))

        # Filter by depth range
        keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (
            np.asarray(pcd.points)[:, 2] <= args.z_far
        )
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        logging.info(f"Point cloud: {len(pcd.points)} points (0 < z <= {args.z_far}m)")

        o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
        rr.log("pcd", rr.Points3D(positions=pcd.points, colors=pcd.colors))
        logging.info(f"Point cloud saved to {args.out_dir}/cloud.ply")

    logging.info("Processing complete!")
