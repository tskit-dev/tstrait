import setuptools
import os

tstrait_version = None
version_file = os.path.join("tstrait", "_version.py")
with open(version_file) as f:
    exec(f.read())

if __name__ == "__main__":
    setuptools.setup(name="tstrait", version=tstrait_version)