import os
import os.path as op
import subprocess

import image_processing

BLACKLIST = ['setup.py', '__init__.py']


def get_python_files():
    """Get all python files from this package
    """
    common_dir = op.abspath(op.dirname(image_processing.__file__))
    flake_check = []
    for dirpath, _, filenames in os.walk(common_dir):
        if dirpath.endswith('tests'):
            continue
        for fn in filenames:
            if op.splitext(fn)[-1].lower() == '.py' and fn not in BLACKLIST:
                path = op.join(dirpath, fn)
                flake_check.append(path)

    return flake_check


def test_code_quality_flake8():
    files = get_python_files()
    command = ['flake8', '--ignore=E131,W503', *[op.abspath(py_file) for py_file in files]]
    subprocess.check_call(command)
