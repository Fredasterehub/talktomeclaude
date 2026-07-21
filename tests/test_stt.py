"""Tests for speech-to-text hardware setup."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from talktomeclaude import stt


class CudaLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        stt._CUDA_DLL_DIRECTORY_HANDLES.clear()
        stt._CUDA_DLL_DIRECTORIES.clear()
        stt._CUDA_DLL_HANDLES.clear()
        stt._CUDA_DLL_PATHS.clear()

    def tearDown(self) -> None:
        stt._CUDA_DLL_DIRECTORY_HANDLES.clear()
        stt._CUDA_DLL_DIRECTORIES.clear()
        stt._CUDA_DLL_HANDLES.clear()
        stt._CUDA_DLL_PATHS.clear()

    def test_windows_registers_nvidia_wheel_bin_directories_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir)
            cublas_bin = package_root / "cublas" / "bin"
            cudnn_bin = package_root / "cudnn" / "bin"
            cublas_bin.mkdir(parents=True)
            cudnn_bin.mkdir(parents=True)
            (cublas_bin / "cublas64_12.dll").touch()
            (cudnn_bin / "cudnn64_9.dll").touch()
            nvidia = SimpleNamespace(__path__=[str(package_root)])
            handles = [object(), object()]
            dll_handles = [object(), object()]

            with mock.patch.object(os, "name", "nt"), mock.patch.dict(
                sys.modules, {"nvidia": nvidia}
            ), mock.patch.dict(
                os.environ, {"PATH": "existing-path"}
            ), mock.patch.object(
                os, "add_dll_directory", side_effect=handles, create=True
            ) as add_directory, mock.patch.object(
                stt.ctypes, "WinDLL", side_effect=dll_handles, create=True
            ) as load_library:
                stt._preload_cuda_libraries()
                stt._preload_cuda_libraries()
                process_path = os.environ["PATH"].split(os.pathsep)

            self.assertEqual(add_directory.call_count, 2)
            registered = {Path(call.args[0]) for call in add_directory.call_args_list}
            self.assertEqual(registered, {cublas_bin, cudnn_bin})
            self.assertEqual(stt._CUDA_DLL_DIRECTORY_HANDLES, handles)
            self.assertEqual(load_library.call_count, 2)
            self.assertEqual(stt._CUDA_DLL_HANDLES, dll_handles)
            self.assertEqual(
                {Path(path) for path in process_path[:2]}, {cublas_bin, cudnn_bin}
            )
            self.assertEqual(process_path[2], "existing-path")

    def test_explicit_cuda_preloads_libraries(self) -> None:
        with mock.patch.object(stt, "_preload_cuda_libraries") as preload:
            tier = stt.detect_tier("cuda")

        preload.assert_called_once_with()
        self.assertEqual(tier, stt.GPU_TIER)


if __name__ == "__main__":
    unittest.main()
