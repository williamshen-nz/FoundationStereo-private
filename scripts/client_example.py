#!/usr/bin/env python3
"""
Example client for the FoundationStereo server.

Usage:
    python client_example.py --left left.png --right right.png
"""

import argparse
import requests
import numpy as np
import io
import cv2


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='http://localhost:1234', help='Server URL')
    parser.add_argument('--left', required=True, help='Left image path')
    parser.add_argument('--right', required=True, help='Right image path')
    parser.add_argument('--fx', type=float, default=754.67, help='Focal length X')
    parser.add_argument('--fy', type=float, default=754.67, help='Focal length Y')
    parser.add_argument('--cx', type=float, default=489.38, help='Principal point X')
    parser.add_argument('--cy', type=float, default=265.16, help='Principal point Y')
    parser.add_argument('--baseline', type=float, default=0.063, help='Baseline in meters')
    parser.add_argument('--format', choices=['npz', 'png'], default='npz', help='Response format')
    parser.add_argument('--output', help='Output depth file (optional)')
    args = parser.parse_args()

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

    if depth is not None and args.output:
        np.save(args.output, depth)
        print(f"Depth saved to {args.output}")
