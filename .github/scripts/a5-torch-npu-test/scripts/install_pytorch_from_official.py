#!/usr/bin/env python3
"""
Install latest PyTorch nightly wheel from official website.
Fallback to a known stable version if fetch fails.
"""

import os
import re
import sys
import subprocess
import urllib.request
from html.parser import HTMLParser
from datetime import datetime


KNOWN_URLS = {
    (2, 13): {
        "aarch64": "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl",
    },
    (2, 14): {
        "aarch64": "https://download-r2.pytorch.org/whl/nightly/cpu/torch-2.14.0.dev20260811%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl",
    },
    (2, 15): {
        "aarch64": "https://download-r2.pytorch.org/whl/nightly/cpu/torch-2.15.0.dev20260816%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl",
        "x86_64": "https://download-r2.pytorch.org/whl/nightly/cpu/torch-2.15.0.dev20260813%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl",
    },
}


class WheelParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for name, value in attrs:
                if name == 'href' and value.endswith('.whl'):
                    self.links.append(value)


def get_version_from_filename(filename):
    m = re.search(r'torch-(\d+)\.(\d+)', filename)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def get_target_version(branch):
    if not branch or branch == 'master' or branch == 'main':
        return None
    m = re.search(r'(\d+)\.(\d+)', branch)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def get_known_url(target_version, arch='aarch64'):
    if target_version and target_version in KNOWN_URLS:
        return KNOWN_URLS[target_version].get(arch)
    return None


def get_latest_url(url, arch='aarch64', python='cp310', variant='cpu', target_version=None):
    print(f"Fetching {url}...")
    
    for domain in ['download-r2.pytorch.org', 'download.pytorch.org']:
        try:
            req_url = url.replace('download.pytorch.org', domain)
            result = subprocess.run(
                ['wget', '-q', '-O', '-', '--no-check-certificate', '--timeout=30', req_url],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0 and result.stdout:
                html = result.stdout
                print(f"Page fetched successfully from {domain}")
                break
        except Exception as e:
            print(f"Failed to fetch from {domain}: {e}")
            continue
    else:
        print("Failed to fetch from all domains (network error)")
        return None

    parser = WheelParser()
    parser.feed(html)
    print(f"Found {len(parser.links)} wheels on page")

    all_versions = set()
    for link in parser.links:
        ver = get_version_from_filename(link)
        if ver:
            all_versions.add(f"{ver[0]}.{ver[1]}")
    
    print(f"Available versions on page: {sorted(all_versions)}")
    
    if target_version:
        version_str = f"{target_version[0]}.{target_version[1]}"
        print(f"Looking for version: {version_str}")
        if version_str not in all_versions:
            print(f"ERROR: Version {version_str} not found on page")
            print(f"ERROR: Available versions: {sorted(all_versions)}")
            return None

    candidates = []
    for link in parser.links:
        if arch not in link or python not in link or variant not in link:
            continue
        if 'linux' not in link and 'manylinux' not in link:
            continue
        
        ver = get_version_from_filename(link)
        if target_version and ver != target_version:
            continue
            
        m = re.search(r'(\d{8})', link)
        if m:
            try:
                date = datetime.strptime(m.group(1), '%Y%m%d')
                candidates.append((date, link))
            except ValueError:
                pass

    if not candidates:
        print(f"No matching wheel found after filtering (arch={arch}, python={python}, variant={variant})")
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    ver = get_version_from_filename(best)
    if ver:
        print(f"Matched PyTorch {ver[0]}.{ver[1]} (latest nightly)")
    
    if best.startswith('http'):
        return best
    return f"https://download.pytorch.org/whl/nightly/torch/{best}"


def get_stable_url(arch='aarch64', python='cp310', variant='cpu', target_version=None):
    if not target_version:
        return None
    
    stable_url = f"https://download.pytorch.org/whl/{variant}/"
    print(f"Trying stable version from {stable_url}...")
    
    for domain in ['download.pytorch.org', 'download-r2.pytorch.org']:
        try:
            req_url = stable_url.replace('download.pytorch.org', domain)
            result = subprocess.run(
                ['wget', '-q', '-O', '-', '--no-check-certificate', '--timeout=60', req_url],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0 and result.stdout:
                html = result.stdout
                break
        except Exception as e:
            continue
    else:
        print("Failed to fetch stable page")
        return None

    parser = WheelParser()
    parser.feed(html)
    print(f"Found {len(parser.links)} wheels in stable")

    candidates = []
    for link in parser.links:
        if arch in link and python in link and variant in link:
            if 'linux' in link or 'manylinux' in link:
                ver = get_version_from_filename(link)
                if ver == target_version:
                    candidates.append(link)

    if not candidates:
        print(f"No stable wheel found for PyTorch {target_version[0]}.{target_version[1]}")
        return None

    print(f"Found stable wheel: {candidates[0]}")
    return candidates[0]


def download(url, output_path):
    print(f"Downloading to {output_path}...")
    
    urls = [url]
    if 'download-r2.pytorch.org' in url:
        urls.append(url.replace('download-r2.pytorch.org', 'download.pytorch.org'))

    for i, u in enumerate(urls):
        if i > 0:
            print("Trying main domain...")
        
        result = subprocess.run(
            ['wget', '--no-check-certificate', '--tries=3', '--retry-connrefused',
             '--timeout=120', '-q', '-O', output_path, u],
            capture_output=True, text=True
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            size = os.path.getsize(output_path) / 1024 / 1024
            print(f"OK ({size:.1f} MB)")
            return True
        
        if os.path.exists(output_path):
            os.remove(output_path)

    return False


def install(wheel_path, python='python3.10'):
    print(f"Installing {os.path.basename(wheel_path)}...")
    result = subprocess.run(
        [python, '-m', 'pip', 'install', '--no-deps', wheel_path],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print("Installed successfully")
        return True
    else:
        print(f"Failed: {result.stderr}")
        return False


def get_filename_from_url(url):
    from urllib.parse import unquote
    filename = url.split('/')[-1]
    return unquote(filename)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='aarch64')
    parser.add_argument('--python-tag', default='cp310')
    parser.add_argument('--variant', default='cpu')
    parser.add_argument('--python-exe', default='python3.10')
    parser.add_argument('--branch', default='master')
    args = parser.parse_args()

    target_version = get_target_version(args.branch)
    ver_str = f"{target_version[0]}.{target_version[1]}" if target_version else "latest"
    print(f"Installing PyTorch {ver_str} (arch={args.arch}, python={args.python_tag})")

    known_url = get_known_url(target_version, args.arch)
    if known_url:
        print(f"Trying hardcoded URL for version {ver_str}...")
        filename = get_filename_from_url(known_url)
        out_path = os.path.join('.', filename)
        if download(known_url, out_path) and install(out_path, args.python_exe):
            verify_and_exit(args.python_exe)
        print("Hardcoded URL failed, crawling page...")

    print("Crawling PyTorch website...")
    latest_url = get_latest_url(
        'https://download.pytorch.org/whl/nightly/torch/',
        args.arch, args.python_tag, args.variant,
        target_version=target_version
    )

    if not latest_url and target_version:
        print("Nightly not found, trying stable...")
        latest_url = get_stable_url(
            args.arch, args.python_tag, args.variant,
            target_version=target_version
        )

    if latest_url:
        filename = get_filename_from_url(latest_url)
        out_path = os.path.join('.', filename)
        print(f"Found: {filename}")
        if download(latest_url, out_path) and install(out_path, args.python_exe):
            verify_and_exit(args.python_exe)

    print("FAILED: Could not install PyTorch")
    sys.exit(1)


def verify_and_exit(python_exe):
    print("Verifying installation...")
    r = subprocess.run(
        [python_exe, '-c', 'import torch; print(torch.__version__)'],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print(f"SUCCESS: PyTorch {r.stdout.strip()} installed")
        sys.exit(0)
    else:
        print("FAILED: Verification failed")
        sys.exit(1)


if __name__ == '__main__':
    main()