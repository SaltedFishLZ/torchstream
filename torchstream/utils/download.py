"""Download API
"""
import shutil
import subprocess

def download_ssh(src, dst):
    subprocess.run(["scp", "-r", src, dst], check=True)

def download_wget(src, dst):
    subprocess.run(["wget", src, dst], check=True)

def download(src, dst, backend="ssh"):
    if backend == "ssh":
        download_ssh(src, dst)
    elif backend == "wget":
        download_wget(src, dst)
    else:
        raise NotImplementedError("Unsupported backend")