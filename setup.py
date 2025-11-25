
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "balacoon_train.data.processors.vc_ops",
        ["src/balacoon_train/data/processors/vc_ops.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    name="balacoon_train",
    version="0.1.0",
    description="Balacoon training utilities",
    author="Balacoon",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "pytorch_lightning",
        "transformers",
        "omegaconf",
        "typing_extensions",
        "numpy",
        "Cython",
    ],
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    extras_require={
        "test": [
            "pytest",
        ],
    },
)
