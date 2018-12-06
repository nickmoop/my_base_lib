import os


def make_directory(directory_path):
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
