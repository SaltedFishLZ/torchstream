"""Download API
"""

import subprocess

def download_ssh(src, dst):
    subprocess.run(["scp", "-r", src, dst], check=True)

def download(src, dst, backend="ssh"):
    if backend == "ssh":
        download_ssh(src, dst)
    else:
        raise NotImplementedError("Unsupported backend")