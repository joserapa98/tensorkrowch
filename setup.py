from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
from pathlib import Path

LIB_NAME = 'tensorkrowch'
version = '1.0.0'

root_dir = Path(LIB_NAME)
include_dir = root_dir / 'include'
ext_src = [str(x.absolute()) for x in root_dir.glob('csrc/*.cpp')]

cpp_extension = CppExtension(
    LIB_NAME + '._C',
    sources=ext_src,
)

setup(
    name=LIB_NAME,
    version=version,
    description='Tensor Networks with PyTorch',
    author='José Ramón Pareja Monturiol, David Pérez-García, Alejandro Pozas-Kerstjens',
    author_email='joserapa@gmail.com',
    license='MIT',
    python_requires='>=3.6',
    install_requires=['torch>=1.0'],
    packages=[LIB_NAME, 'tensorkrowch.models'],
    ext_modules=[cpp_extension],
    cmdclass={'build_ext': BuildExtension},
)