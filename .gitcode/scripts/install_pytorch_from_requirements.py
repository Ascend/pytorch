#!/usr/bin/env python3
"""按 PR build_job.yml 的方式安装 PyTorch: 从 requirements.txt 读版本并走 OBS 源。"""

import os
import re
import sys
import subprocess


def install_from_requirements(python_exe='python3.10', arch='aarch64', python_tag='cp310'):
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'requirements.txt')
    torch_version = None
    with open(req_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('torch=='):
                torch_version = line[len('torch=='):].strip()
                break
    if not torch_version:
        print("FAILED: torch== not found in requirements.txt")
        return False

    base_version = torch_version.split('+')[0]      # e.g. 2.15.0.dev20260902
    date_match = re.search(r'(\d{8})$', base_version)
    if date_match:
        ver = base_version
        url_base = ("https://pytorch-package.obs.cn-north-4.myhuaweicloud.com"
                    f"/pta/torch/master/{date_match.group(1)}")
    else:
        ver = base_version
        url_base = ("https://pytorch-package.obs.cn-north-4.myhuaweicloud.com"
                    f"/pta/torch/v{ver}")

    if arch in ('x64', 'x86', 'x86_64'):
        torch_arch = 'x86_64'
    elif arch in ('arm64', 'arm'):
        torch_arch = 'aarch64'
    else:
        torch_arch = arch

    torch_url = (f"{url_base}/torch-{ver}%2Bcpu-{python_tag}-{python_tag}"
                 f"-manylinux_2_28_{torch_arch}.whl")
    print(f"torch_url={torch_url}")

    print(f"Installing via pip: {torch_url}")
    result = subprocess.run([python_exe, '-m', 'pip', 'install', torch_url])
    return result.returncode == 0


def verify(python_exe):
    r = subprocess.run(
        [python_exe, '-m', 'pip', 'show', 'torch'],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print("FAILED: pip show torch failed")
        return False
    print("SUCCESS: PyTorch installed")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='aarch64')
    parser.add_argument('--python-tag', default='cp310')
    parser.add_argument('--python-exe', default='python3.10')
    args = parser.parse_args()

    if not install_from_requirements(args.python_exe, args.arch, args.python_tag):
        print("FAILED: Could not install PyTorch")
        sys.exit(1)
    if not verify(args.python_exe):
        sys.exit(1)


if __name__ == '__main__':
    main()
