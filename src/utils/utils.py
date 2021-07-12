import os
import glob


def get_files(dir, suffix='.bvh', files=[]):
    files += glob.glob(os.path.join(dir, '*'+suffix))
    for subdir in next(os.walk(dir))[1]:
        get_files(os.path.join(dir, subdir), suffix, files)