"""
Convert custom_Dataset (29 input views + 2 target views) into a DepthSplat .torch chunk.

Input data layout:
  custom_Dataset/inputs/rgb_0.png .. rgb_28.png   (29 input images, 1440x1080)
  custom_Dataset/inputs/metadata.json             (camera_to_world in OpenCV c2w, camera_to_pixel intrinsics)
  custom_Dataset/outputs/cameras.json             (2 target camera poses in Blender/z_back convention)

Output:
  depthsplat/datasets/custom/test/000000.torch    (single chunk with 31 views: 29 context + 2 target)
  depthsplat/datasets/custom/test/index.json
  depthsplat/assets/custom_eval_index.json        (evaluation index: context=0..28, target=29..30)
"""

import json
import struct
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

CUSTOM_DIR = Path(__file__).resolve().parent.parent / "custom_Dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "datasets" / "custom" / "test"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def load_raw(path: Path) -> torch.Tensor:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def blender_to_opencv_c2w(c2w_blender: list[list[float]]) -> np.ndarray:
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
    )
    return np.array(c2w_blender, dtype=np.float64) @ blender2opencv


def main():
    with open(CUSTOM_DIR / "inputs" / "metadata.json") as f:
        meta = json.load(f)

    with open(CUSTOM_DIR / "outputs" / "cameras.json") as f:
        out_cams = json.load(f)

    cam = meta["camera"]
    num_input = len(cam["camera_to_world"])  # 29
    num_target = len(out_cams["camera_to_world"])  # 2
    num_total = num_input + num_target
    print(f"Input views: {num_input}, Target views: {num_target}, Total: {num_total}")

    img_w, img_h = cam["image_size_xy"][0]
    img_w, img_h = int(img_w), int(img_h)
    print(f"Image resolution: {img_w}x{img_h} (WxH), i.e. {img_h}x{img_w} (HxW)")

    # ---- Build camera tensor [N, 18] ----
    # Format: [fx_norm, fy_norm, cx_norm, cy_norm, 0, 0, w2c[:3].flatten() (12 values)]
    K = cam["camera_to_pixel"][0]
    fx, fy = K[0][0], K[1][1]
    cx, cy = K[0][2], K[1][2]
    fx_norm = fx / img_w
    fy_norm = fy / img_h
    cx_norm = cx / img_w
    cy_norm = cy / img_h
    print(f"Normalized intrinsics: fx={fx_norm:.6f}, fy={fy_norm:.6f}, cx={cx_norm:.6f}, cy={cy_norm:.6f}")

    cameras = []

    # Input views: c2w already in OpenCV convention (metadata says so)
    for i in range(num_input):
        c2w = np.array(cam["camera_to_world"][i], dtype=np.float64)
        w2c = np.linalg.inv(c2w)
        cam_vec = [fx_norm, fy_norm, cx_norm, cy_norm, 0.0, 0.0]
        cam_vec.extend(w2c[:3].flatten().tolist())
        cameras.append(cam_vec)

    # Target views: cameras.json uses Blender/z_back convention, convert to OpenCV
    for i in range(num_target):
        c2w_opencv = blender_to_opencv_c2w(out_cams["camera_to_world"][i])
        w2c = np.linalg.inv(c2w_opencv)
        cam_vec = [fx_norm, fy_norm, cx_norm, cy_norm, 0.0, 0.0]
        cam_vec.extend(w2c[:3].flatten().tolist())
        cameras.append(cam_vec)

    cameras_tensor = torch.tensor(np.array(cameras), dtype=torch.float32)
    print(f"Cameras tensor shape: {cameras_tensor.shape}")

    # ---- Load images as raw bytes ----
    images = []
    input_dir = CUSTOM_DIR / "inputs"
    for i in range(num_input):
        img_path = input_dir / f"rgb_{i}.png"
        if not img_path.exists():
            print(f"ERROR: {img_path} not found!")
            sys.exit(1)
        images.append(load_raw(img_path))
    print(f"Loaded {num_input} input images")

    # Target views need placeholder images (model renders these, originals only
    # used for GT comparison / metrics). Create black PNGs at the same resolution.
    for i in range(num_target):
        buf = BytesIO()
        PILImage.new("RGB", (img_w, img_h), (0, 0, 0)).save(buf, format="PNG")
        buf.seek(0)
        images.append(torch.tensor(np.frombuffer(buf.read(), dtype=np.uint8)))
    print(f"Total images (input + placeholder targets): {len(images)}")

    # ---- Build timestamps ----
    timestamps = torch.arange(num_total, dtype=torch.int64)

    # ---- Create chunk ----
    example = {
        "key": "custom_scene",
        "url": "custom",
        "timestamps": timestamps,
        "cameras": cameras_tensor,
        "images": images,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_path = OUTPUT_DIR / "000000.torch"
    torch.save([example], chunk_path)
    print(f"\nSaved chunk: {chunk_path}")

    # ---- Create index.json ----
    index = {"custom_scene": "000000.torch"}
    with open(OUTPUT_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Saved index:  {OUTPUT_DIR / 'index.json'}")

    # ---- Create evaluation index ----
    context_indices = list(range(num_input))
    target_indices = list(range(num_input, num_total))

    eval_index = {
        "custom_scene": {
            "context": context_indices,
            "target": target_indices,
        }
    }

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = ASSETS_DIR / "custom_eval_index.json"
    with open(eval_path, "w") as f:
        json.dump(eval_index, f, indent=2)
    print(f"Saved eval index: {eval_path}")
    print(f"  context indices: {context_indices}")
    print(f"  target indices:  {target_indices}")

    print("\n" + "=" * 70)
    print("DONE! Now run DepthSplat inference (from the depthsplat/ directory).")
    print("=" * 70)
    print()
    print("Step 1: Download the pretrained model (if not already done):")
    print("  mkdir -p pretrained")
    print("  wget https://huggingface.co/haofeixu/depthsplat/resolve/main/"
          "depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth "
          "-P pretrained")
    print()
    print("Step 2: Run inference:")
    print(
        "CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv \\\n"
        "  dataset.roots=[datasets/custom] \\\n"
        "  dataset.test_chunk_interval=1 \\\n"
        "  dataset.image_shape=[512,960] \\\n"
        "  dataset.ori_image_shape=[1080,1440] \\\n"
        "  model.encoder.upsample_factor=8 \\\n"
        "  model.encoder.lowest_feature_resolution=8 \\\n"
        "  model.encoder.gaussian_adapter.gaussian_scale_max=0.1 \\\n"
        "  checkpointing.pretrained_model=pretrained/depthsplat-gs-small-re10kdl3dv-448x768-randview4-10-c08188db.pth \\\n"
        "  mode=test \\\n"
        "  dataset/view_sampler=evaluation \\\n"
        "  dataset.view_sampler.num_context_views=29 \\\n"
        "  dataset.view_sampler.index_path=assets/custom_eval_index.json \\\n"
        "  test.save_image=true \\\n"
        "  test.compute_scores=false \\\n"
        "  output_dir=outputs/custom_scene"
    )


if __name__ == "__main__":
    main()
