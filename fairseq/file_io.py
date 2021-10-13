#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from logging import log
import os
import re
import shutil
from typing import List, Optional
import tensorflow.io.gfile as gf
import time
import logging


logger = logging.getLogger(__file__)

class PathManager:
    """
    Wrapper for insulating OSS I/O (using Python builtin operations) from
    fvcore's PathManager abstraction (for transparently handling various
    internal backends).
    """

    @staticmethod
    def open(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        if path.startswith("hdfs://"):
            return gf.GFile(path, mode=mode)
        return open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    @staticmethod
    def copy(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        if src_path.startswith("hdfs://"):
            gf.copy(src_path, dst_path, overwrite)
            return True
        return shutil.copyfile(src_path, dst_path)

    @staticmethod
    def get_local_path(path: str, **kwargs) -> str:
        # if path.startswith("hdfs://"):
        #     local_path = "/tmp/{}".format(time.time())
        #     gf.copy(path, local_path)
        #     logger.info("load {} to {}".format(path, local_path))
        #     return local_path
        return path

    @staticmethod
    def exists(path: str) -> bool:
        if path.startswith("hdfs://"):
            return gf.exists(path)
        return os.path.exists(path)

    @staticmethod
    def lexists(path: str) -> bool:
        if path.startswith("hdfs://"):
            return gf.exists(path)   # --> hack
        return os.path.lexists(path)


    @staticmethod
    def isfile(path: str) -> bool:
        if path.startswith("hdfs://"):
            return PathManager.exists(path)
        return os.path.isfile(path)

    @staticmethod
    def ls(path: str) -> List[str]:
        if path.startswith("hdfs://"):
            return gf.listdir(path)
        return os.listdir(path)

    @staticmethod
    def supports_rename(path: str) -> bool:
        return True                  # --> hack

    @staticmethod
    def mkdirs(path: str) -> None:
        if path.startswith("hdfs://"):
            if not PathManager.exists(path):
                gf.makedirs(path)
            return
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rm(path: str) -> None:
        if path.startswith("hdfs://"):
            gf.remove(path)
            return
        os.remove(path)

    @staticmethod
    def chmod(path: str, mode: int) -> None:
        os.chmod(path, mode)

    @staticmethod
    def copy_from_local(
        local_path: str, dst_path: str, overwrite: bool = False, **kwargs
    ) -> None:
        return shutil.copyfile(local_path, dst_path)

    @staticmethod
    def rename(src: str, dst: str):
        if src.startswith("hdfs://"):
            gf.rename(src, dst)
            return
        os.rename(src, dst)

    @staticmethod
    def join_path(a, *paths):
        return os.path.join(a, *paths)
