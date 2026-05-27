import argparse
from huggingface_hub import snapshot_download

def parse_args():
    p = argparse.ArgumentParser(description="Download a Hugging Face repo snapshot.")
    p.add_argument(
        "--model", "--repo_id",
        dest="repo_id",
        type=str,
        required=True,
        help='Hugging Face repo id, e.g. "openai/gpt-oss-20b"'
    )
    p.add_argument(
        "--save_path", "--local_dir",
        dest="save_dir",
        type=str,
        default="./gptoss",
        help='Local directory to save files (default: "./gptoss")'
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help='Cache directory (default: <save_path>/cache)'
    )
    p.add_argument(
        "--no_symlinks",
        action="store_true",
        help="Do not use symlinks (equivalent to local_dir_use_symlinks=False)."
    )
    p.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume download."
    )
    return p.parse_args()

def main():
    args = parse_args()

    cache_dir = args.cache_dir or (args.save_dir.rstrip("/") + "/cache")

    snapshot_download(
        repo_id=args.repo_id,
        cache_dir=cache_dir,
        local_dir=args.save_dir,
        local_dir_use_symlinks=not args.no_symlinks,
        resume_download=not args.no_resume,
    )

if __name__ == "__main__":
    main()