import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(base_path)


def get_url(name):
    if not name:
        return ""
    path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path, "url.ini"), "r") as f:
        content = f.read()
        if name not in content:
            return ""
        _url = content.split(name + "=")[1].split('\n')[0].strip()
    return _url
