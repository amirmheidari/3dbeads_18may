import subprocess
import sys
import time


def test_smoke_runs_quickly():
    start = time.time()
    subprocess.check_call([sys.executable, 'train.py', '--smoke', '--iters', '1'])
    assert time.time() - start < 5

