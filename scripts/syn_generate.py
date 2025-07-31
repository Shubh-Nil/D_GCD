#!/usr/bin/env python
import argparse
import os
from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from tqdm import tqdm

def load_image(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

def main():
    parser = argparse.ArgumentParser(
        description="Batch apply Instruct-Pix2Pix prompt to a folder of images."
    )
    parser.add_argument(
        "--source_dir", "-s", required=True,
        help="Root folder of your dataset (e.g. datasets/PACS)"
    )
    parser.add_argument(
        "--target_dir", "-t", required=True,
        help="Where to write the transformed images"
    )
    parser.add_argument(
        "--prompt", "-p", required=True,
        help="Text instruction, e.g. 'Add snow background'"
    )
    parser.add_argument(
        "--model_id", default="timbrooks/instruct-pix2pix",
        help="Diffusers model to use"
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Torch device"
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=1.0,
        help="Image guidance scale"
    )
    args = parser.parse_args()

    # 1) Load the pipeline
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    # 2) Walk the source tree, process every image
    for root, _, files in os.walk(args.source_dir):
        # preserve sub-folder structure
        rel_folder = os.path.relpath(root, args.source_dir)
        out_folder = os.path.join(args.target_dir, rel_folder)
        os.makedirs(out_folder, exist_ok=True)

        for fname in tqdm(files, desc=rel_folder or "root"):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            src_path = os.path.join(root, fname)
            dst_path = os.path.join(out_folder, fname)
            if os.path.exists(dst_path):
                # skip if already done
                continue

            try:
                img = load_image(src_path)
                out_imgs = pipe(
                    args.prompt,
                    image=img,
                    num_inference_steps=args.steps,
                    image_guidance_scale=args.guidance_scale
                ).images
                out_imgs[0].save(dst_path)
                # free GPU memory
                del out_imgs
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"[ERROR] {src_path} → {e}")

    print(f"\Completed!")

if __name__ == "__main__":
    main()
