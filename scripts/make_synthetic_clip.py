"""Write a procedural green/blue-screen test clip with its exact ground truth.

    python scripts/make_synthetic_clip.py out_dir --frames 48 --width 1920 --height 1080

Creates out_dir/frames/*.png (the plate), out_dir/alpha/*.png (true matte) and
out_dir/foreground/*.png (true straight foreground). Handy for checking a
keying setup or as fine-tuning targets.
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("ck2pack")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("ck2pack", package)

from ck2pack.ck_comfy.synthetic import make_clip  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--screen", choices=("green", "blue"), default="green")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clip = make_clip(args.frames, args.width, args.height, seed=args.seed, screen=args.screen)
    out = Path(args.out_dir)
    for name, key in (("frames", "images"), ("alpha", "alpha"), ("foreground", "fg")):
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        for t, frame in enumerate(clip[key]):
            data = (frame.numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(data).save(folder / f"{t:05d}.png")
    print(f"wrote {args.frames} frames to {out}")


if __name__ == "__main__":
    main()
