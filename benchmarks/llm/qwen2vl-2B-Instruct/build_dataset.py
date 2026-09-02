#!/usr/bin/env python3
import io
import json
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
import webdataset as wds
from PIL import Image


USER_PROMPT_DEFAULT = "请描述这张图片的内容。"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pick_caption(sample: dict):
    for k in ("txt", "text", "caption", "captions", "description"):
        if k in sample and sample[k] is not None:
            v = sample[k]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="ignore")
            if isinstance(v, str):
                v = v.strip()
                if v:
                    return v

    for k in ("json", "meta", "metadata"):
        if k in sample and sample[k] is not None:
            v = sample[k]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="ignore")
            if isinstance(v, str):
                v = v.strip()
                if not v:
                    continue
                try:
                    obj = json.loads(v)
                    for kk in ("caption", "text", "description"):
                        if kk in obj and isinstance(obj[kk], str) and obj[kk].strip():
                            return obj[kk].strip()
                except Exception:
                    pass

    return None


def save_image_bytes(img_bytes: bytes, out_path: Path):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.save(out_path, format="JPEG", quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="pixparse/cc12m-wds")
    ap.add_argument("--rev", default="main")
    ap.add_argument("--pattern", action="append", default=["cc12m-train-0000.tar"],
                    help="Repeatable. e.g. --pattern 'cc12m-train-000*.tar'")
    ap.add_argument("--out-json", default="train_data_0.json")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--local-dir", default="./cc12m-wds-snapshot")
    ap.add_argument("--img-dir", default="./cc12m_dataset_images",
                    help="Where to write extracted images. Set empty to skip saving images.")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="0 means no limit.")
    ap.add_argument("--user-prompt", default=USER_PROMPT_DEFAULT)
    ap.add_argument("--token", default=None, help="HF token if needed.")
    args = ap.parse_args()

    snapshot_path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.rev,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        allow_patterns=args.pattern,
        token=args.token,
    )

    local_dir = Path(args.local_dir)
    tar_paths = []
    for pat in args.pattern:
        tar_paths.extend(sorted(local_dir.glob(pat)))
    tar_paths = sorted(set(tar_paths))

    if not tar_paths:
        raise FileNotFoundError(
            f"No files matching {args.pattern} found in {local_dir}."
        )

    img_dir = None
    if args.img_dir and args.img_dir.strip():
        img_dir = Path(args.img_dir)
        ensure_dir(img_dir)

    results = []
    idx = 0

    dataset = wds.WebDataset([str(p) for p in tar_paths]).decode()

    for sample in dataset:
        if args.max_samples and idx >= args.max_samples:
            break

        img_bytes = None
        img_ext = None
        for k in ("jpg", "jpeg", "png", "webp"):
            if k in sample and sample[k] is not None:
                v = sample[k]
                if isinstance(v, bytes):
                    img_bytes = v
                else:
                    try:
                        buf = io.BytesIO()
                        v.convert("RGB").save(buf, format="JPEG", quality=95)
                        img_bytes = buf.getvalue()
                    except Exception:
                        img_bytes = None
                img_ext = "jpg"
                break

        if img_bytes is None:
            continue

        caption = pick_caption(sample)
        if not caption:
            continue

        key = sample.get("__key__", f"{idx:08d}")
        img_name = f"{key}.{img_ext}"

        if img_dir is not None:
            out_img_path = img_dir / img_name
            if not out_img_path.exists():
                try:
                    save_image_bytes(img_bytes, out_img_path)
                except Exception:
                    continue

        item = {
            "id": str(idx),
            "image": img_name if img_dir is not None else img_name,
            "conversations": [
                {"role": "user", "content": args.user_prompt},
                {"role": "assistant", "content": caption.strip()},
            ],
        }
        results.append(item)
        idx += 1

        if idx % 1000 == 0:
            print(f"processed: {idx}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done: {len(results)} items total, JSON: {args.out_json}")
    if img_dir is not None:
        print(f"Image save directory: {img_dir.resolve()}")
    print(f"Using shards: {', '.join([p.name for p in tar_paths])}")
    print(f"Snapshot path: {snapshot_path}")


if __name__ == "__main__":
    main()
