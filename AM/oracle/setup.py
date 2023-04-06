#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension(
            "oracle",
            sources = [
                "oracle.pyx",
                "cpp_oracle.cpp",
                "common.cpp",
                "Conflict.cpp",
                "ConstraintTable.cpp",
                "Instance.cpp",
                "PBS.cpp",
                "PBSNode.cpp",
                "ReservationTable.cpp",
                "SingleAgentSolver.cpp",
                "SIPP.cpp",
            ],
            language="c++",
            extra_compile_args = [
                "-w",
                "-Wno-everything",
                "-Ofast",
                "-march=native",
                "-funroll-loops",
                "-funsafe-loop-optimizations",
                "-falign-functions=16",
                "-falign-loops=16"
            ],
            include_dirs = [
                numpy.get_include()
            ]
        )
    ]
)
