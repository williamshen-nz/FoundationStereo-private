"""
Standalone client for FoundationStereo server.

This file can be copied to any project that needs to use the stereo depth server.

Dependencies:
    pip install requests numpy opencv-python

Example usage:
    from stereo_client import StereoClient

    client = StereoClient("http://localhost:8000")

    # From file paths
    depth = client.infer_from_files("left.png", "right.png",
                                     fx=754.67, fy=754.67,
                                     cx=489.38, cy=265.16,
                                     baseline=0.063)

    # From numpy arrays
    depth = client.infer_from_arrays(left_rgb, right_rgb, K, baseline)
"""

import io
import requests
import numpy as np
import cv2
from typing import Optional, Tuple


class StereoClient:
    """Client for FoundationStereo depth inference server"""

    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        Initialize client.

        Args:
            server_url: Base URL of the server (e.g., "http://localhost:8000")
        """
        self.server_url = server_url.rstrip('/')

    def health_check(self) -> bool:
        """
        Check if server is healthy and model is loaded.

        Returns:
            True if server is ready, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=2)
            if response.status_code != 200:
                return False
            data = response.json()
            return data.get('model_loaded', False)
        except Exception:
            return False

    def infer_from_files(self,
                        left_path: str,
                        right_path: str,
                        fx: float,
                        fy: float,
                        cx: float,
                        cy: float,
                        baseline: float,
                        scale: float = 1.0,
                        hiera: bool = False,
                        valid_iters: int = 32,
                        format: str = 'npz') -> Tuple[np.ndarray, float]:
        """
        Infer depth from image files.

        Args:
            left_path: Path to left image
            right_path: Path to right image
            fx, fy, cx, cy: Camera intrinsics
            baseline: Stereo baseline in meters
            scale: Image downscale factor (default: 1.0)
            hiera: Use hierarchical inference (default: False)
            valid_iters: Number of iterations (default: 32)
            format: 'npz' or 'png' (default: 'npz')

        Returns:
            (depth, inference_time) where depth is (H, W) float32 array in meters
        """
        with open(left_path, 'rb') as f:
            left_bytes = f.read()
        with open(right_path, 'rb') as f:
            right_bytes = f.read()

        return self._infer(left_bytes, right_bytes, fx, fy, cx, cy, baseline,
                          scale, hiera, valid_iters, format)

    def infer_from_arrays(self,
                         left_rgb: np.ndarray,
                         right_rgb: np.ndarray,
                         K: np.ndarray,
                         baseline: float,
                         scale: float = 1.0,
                         hiera: bool = False,
                         valid_iters: int = 32,
                         format: str = 'npz') -> Tuple[np.ndarray, float]:
        """
        Infer depth from RGB numpy arrays.

        Args:
            left_rgb: Left image (H, W, 3) RGB uint8
            right_rgb: Right image (H, W, 3) RGB uint8
            K: Camera intrinsic matrix (3, 3) [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            baseline: Stereo baseline in meters
            scale: Image downscale factor (default: 1.0)
            hiera: Use hierarchical inference (default: False)
            valid_iters: Number of iterations (default: 32)
            format: 'npz' or 'png' (default: 'npz')

        Returns:
            (depth, inference_time) where depth is (H, W) float32 array in meters
        """
        # Encode images as PNG bytes
        left_bgr = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR)

        left_success, left_buffer = cv2.imencode('.png', left_bgr)
        right_success, right_buffer = cv2.imencode('.png', right_bgr)

        if not left_success or not right_success:
            raise ValueError("Failed to encode images")

        left_bytes = left_buffer.tobytes()
        right_bytes = right_buffer.tobytes()

        return self._infer(left_bytes, right_bytes,
                          float(K[0, 0]), float(K[1, 1]),
                          float(K[0, 2]), float(K[1, 2]),
                          float(baseline),
                          scale, hiera, valid_iters, format)

    def _infer(self,
              left_bytes: bytes,
              right_bytes: bytes,
              fx: float, fy: float, cx: float, cy: float,
              baseline: float,
              scale: float,
              hiera: bool,
              valid_iters: int,
              format: str) -> Tuple[np.ndarray, float]:
        """Internal method to send request and parse response"""

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
            'scale': scale,
            'hiera': int(hiera),
            'valid_iters': valid_iters,
        }

        # Choose endpoint
        endpoint = f"{self.server_url}/infer" if format == 'npz' else f"{self.server_url}/infer_png"

        # Send request
        response = requests.post(endpoint, files=files, data=data)

        if response.status_code != 200:
            raise RuntimeError(f"Server error {response.status_code}: {response.text}")

        # Get inference time from headers
        inference_time = float(response.headers.get('X-Inference-Time', 0))

        # Decode depth based on format
        if format == 'npz':
            depth = np.load(io.BytesIO(response.content))['depth']
        else:
            # Decode PNG
            nparr = np.frombuffer(response.content, np.uint8)
            depth_mm = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            depth_scale = float(response.headers.get('X-Depth-Scale', 1000.0))
            depth = depth_mm.astype(np.float32) / depth_scale
            depth[depth_mm == 0] = np.nan

        return depth, inference_time


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Standalone stereo client example")
    parser.add_argument('--server', default='http://localhost:8000', help='Server URL')
    parser.add_argument('--left', required=True, help='Left image path')
    parser.add_argument('--right', required=True, help='Right image path')
    parser.add_argument('--fx', type=float, default=754.67, help='Focal length X')
    parser.add_argument('--fy', type=float, default=754.67, help='Focal length Y')
    parser.add_argument('--cx', type=float, default=489.38, help='Principal point X')
    parser.add_argument('--cy', type=float, default=265.16, help='Principal point Y')
    parser.add_argument('--baseline', type=float, default=0.063, help='Baseline in meters')
    parser.add_argument('--format', choices=['npz', 'png'], default='npz', help='Response format')
    parser.add_argument('--output', help='Output depth file (.npy)')
    args = parser.parse_args()

    # Create client
    client = StereoClient(args.server)

    # Check health
    if not client.health_check():
        print(f"ERROR: Server at {args.server} is not healthy!")
        print("Make sure server is running: python scripts/server.py")
        exit(1)

    print(f"Server is healthy at {args.server}")
    print(f"Sending stereo pair for inference (format: {args.format})...")

    # Infer depth
    depth, inference_time = client.infer_from_files(
        args.left, args.right,
        args.fx, args.fy, args.cx, args.cy, args.baseline,
        format=args.format
    )

    # Print stats
    valid_depth = depth[np.isfinite(depth)]
    print(f"Inference time: {inference_time:.3f}s")
    print(f"Depth shape: {depth.shape}")
    if len(valid_depth) > 0:
        print(f"Depth range: {valid_depth.min():.2f}m - {valid_depth.max():.2f}m")
        print(f"Mean depth: {valid_depth.mean():.2f}m")

    # Save if requested
    if args.output:
        np.save(args.output, depth)
        print(f"Depth saved to {args.output}")
