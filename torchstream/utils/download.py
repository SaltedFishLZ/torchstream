"""Download API
For directory, you need a trailing `/`
"""
import shutil
import subprocess

def download_ssh(src, dst):
    subprocess.run(["scp", "-r", src, dst], check=True)

def download_wget(src, dst):
    subprocess.run(["wget", src, dst], check=True)

def download_rsync(src, dst):
    subprocess.run(["rsync", "-ur", src, dst], check=True)

def download(src, dst, backend="ssh"):
    if backend == "ssh":
        download_ssh(src, dst)
    elif backend == "wget":
        download_wget(src, dst)
    elif backend == "rsync":
        download_rsync(src, dst)
    else:
        raise NotImplementedError("Unsupported backend")
