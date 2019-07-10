# -*- coding: utf-8 -*-

import os
import platform
import struct
import sys

python_version_major = sys.version_info.major

if python_version_major == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve


def download_spade_fim():
    """
    Downloads SPADE specific PyFIM binary file.
    """
    if platform.system() == "Windows":
        fim_filename = "fim.pyd"
    else:
        # Linux
        fim_filename = "fim.so"
    root_dir = os.path.dirname(os.path.dirname(__file__))
    spade_src_dir = os.path.join(root_dir, "elephant", "spade_src")
    fim_lib_path = os.path.join(spade_src_dir, fim_filename)
    if os.path.exists(fim_lib_path):
        return

    arch_bits = struct.calcsize("P") * 8
    url_fim = "http://www.borgelt.net/bin{arch}/py{py_ver}/{filename}". \
        format(arch=arch_bits, py_ver=python_version_major,
               filename=fim_filename)
    try:
        urlretrieve(url_fim, filename=fim_lib_path)
        print("Successfully downloaded fim lib to {}".format(fim_lib_path))
    except Exception:
        print("Unable to download {url} module.".format(url=url_fim))
