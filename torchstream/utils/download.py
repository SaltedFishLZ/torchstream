"""Download API
"""

import subprocess

def download_ssh(src, dst):
    subprocess.run(["scp", src, dst])