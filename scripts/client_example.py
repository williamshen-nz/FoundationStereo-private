#!/usr/bin/env python3
"""
Example client for the FoundationStereo server.

Usage:
    python client_example.py

    Or with custom images:
    python client_example.py --left left.png --right right.png
"""

import os
import argparse
import requests
import numpy as np
import io
import cv2


def is_display_available():
    """Check if display is available for visualization."""
    import os
    import platform

    # Check DISPLAY environment variable on Linux/Unix
    if platform.system() in ['Linux', 'Darwin']:
        if not os.environ.get('DISPLAY'):
            return False

    # Try to initialize a window
    try:
        test_img = np.zeros((1, 1, 3), dtype=np.uint8)
        cv2.imshow('test', test_img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except:
        return False


def visualize_depth(depth, window_name="Depth Map"):
    """Visualize depth map with colormap. Press 'q' to close."""
    # Check if display is available
    if not is_display_available():
        print("\nNo display available - skipping visualization")
        print("(Running headless or DISPLAY not set)")
        return

    # Create a copy to avoid modifying original
    depth_vis = depth.copy()

    # Handle NaN and inf values
    valid_mask = np.isfinite(depth_vis)

    if valid_mask.sum() == 0:
        print("No valid depth values to visualize")
        return

    # Normalize to 0-255 for visualization
    min_val = np.nanmin(depth_vis[valid_mask])
    max_val = np.nanmax(depth_vis[valid_mask])

    depth_normalized = (depth_vis - min_val) / (max_val - min_val)
    depth_normalized = np.nan_to_num(depth_normalized, nan=0.0)
    depth_uint8 = (depth_normalized * 255).clip(0, 255).astype(np.uint8)

    # Apply colormap (TURBO for better visualization)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

    # Set invalid pixels to black
    depth_colored[~valid_mask] = 0

    # Display
    cv2.imshow(window_name, depth_colored)
    print(f"\nDisplaying depth map. Press 'q' to close the window.")
    print(f"Depth range: {min_val:.2f}m - {max_val:.2f}m")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break

    cv2.destroyAllWindows()


def infer_depth_npz(server_url, left_path, right_path, fx, fy, cx, cy, baseline):
    """Send images to server and get depth as NPZ"""

    with open(left_path, 'rb') as f:
        left_bytes = f.read()
    with open(right_path, 'rb') as f:
        right_bytes = f.read()

    files = {
        'left_image': ('left.png', left_bytes, 'image/png'),
        'right_image': ('right.png', right_bytes, 'image/png'),
    }

    data = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'baseline': baseline,
        'scale': 1.0,
        'hiera': 0,
        'valid_iters': 32
    }

    response = requests.post(f"{server_url}/infer", files=files, data=data)

    if response.status_code == 200:
        # Load depth from NPZ
        buffer = io.BytesIO(response.content)
        depth = np.load(buffer)['depth']

        print(f"Inference time: {response.headers.get('X-Inference-Time')}s")
        print(f"Depth shape: {depth.shape}")
        print(f"Depth range: {np.nanmin(depth):.2f}m - {np.nanmax(depth):.2f}m")

        return depth
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def infer_depth_png(server_url, left_path, right_path, fx, fy, cx, cy, baseline):
    """Send images to server and get depth as 16-bit PNG"""

    with open(left_path, 'rb') as f:
        left_bytes = f.read()
    with open(right_path, 'rb') as f:
        right_bytes = f.read()

    files = {
        'left_image': ('left.png', left_bytes, 'image/png'),
        'right_image': ('right.png', right_bytes, 'image/png'),
    }

    data = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'baseline': baseline,
        'scale': 1.0,
        'hiera': 0,
        'valid_iters': 32,
        'depth_scale': 1000.0  # millimeters
    }

    response = requests.post(f"{server_url}/infer_png", files=files, data=data)

    if response.status_code == 200:
        # Decode PNG
        nparr = np.frombuffer(response.content, np.uint8)
        depth_mm = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        # Convert back to meters
        depth_scale = float(response.headers.get('X-Depth-Scale', 1000.0))
        depth = depth_mm.astype(np.float32) / depth_scale
        depth[depth_mm == 0] = np.nan

        print(f"Inference time: {response.headers.get('X-Inference-Time')}s")
        print(f"Depth shape: {depth.shape}")
        print(f"Depth range: {np.nanmin(depth):.2f}m - {np.nanmax(depth):.2f}m")

        return depth
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


if __name__ == "__main__":
    # Get default paths to example data
    code_dir = os.path.dirname(os.path.realpath(__file__))
    default_left = os.path.join(code_dir, '..', 'assets', 'left.png')
    default_right = os.path.join(code_dir, '..', 'assets', 'right.png')

    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='http://localhost:1234', help='Server URL')
    parser.add_argument('--left', default=default_left, help='Left image path (default: assets/left.png)')
    parser.add_argument('--right', default=default_right, help='Right image path (default: assets/right.png)')
    parser.add_argument('--fx', type=float, default=754.6680908203125, help='Focal length X')
    parser.add_argument('--fy', type=float, default=754.6680908203125, help='Focal length Y')
    parser.add_argument('--cx', type=float, default=489.3794860839844, help='Principal point X')
    parser.add_argument('--cy', type=float, default=265.16162109375, help='Principal point Y')
    parser.add_argument('--baseline', type=float, default=0.063, help='Baseline in meters')
    parser.add_argument('--format', choices=['npz', 'png'], default='npz', help='Response format')
    parser.add_argument('--output', help='Output depth file (optional)')
    args = parser.parse_args()

    print(f"Using left image: {args.left}")
    print(f"Using right image: {args.right}")
    print(f"Camera intrinsics: fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}, baseline={args.baseline}m")
    print(f"Sending request to {args.server}...")

    if args.format == 'npz':
        depth = infer_depth_npz(
            args.server, args.left, args.right,
            args.fx, args.fy, args.cx, args.cy, args.baseline
        )
    else:
        depth = infer_depth_png(
            args.server, args.left, args.right,
            args.fx, args.fy, args.cx, args.cy, args.baseline
        )

    if depth is not None:
        # Visualize depth
        visualize_depth(depth)

        # Save if output path specified
        if args.output:
            np.save(args.output, depth)
            print(f"Depth saved to {args.output}")
