import os

def check_and_make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
