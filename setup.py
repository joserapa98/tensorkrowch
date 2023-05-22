from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
from pathlib import Path

LIB_NAME = 'tensorkrowch'

# Version
with open('tensorkrowch/_version.py') as f:
    exec(f.read())  # __version__

# Pytorch Cpp Extension
cpp_extension = CppExtension(LIB_NAME + '._C', sources=['tensorkrowch/csrc/operations.cpp'])


setup(
    name=LIB_NAME,
    version=__version__,
    description='Tensor Networks with PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/joserapa98/tensorkrowch',
    project_urls={'Documentation': 'https://joserapa98.github.io/tensorkrowch'},
    author='José Ramón Pareja Monturiol, David Pérez-García, Alejandro Pozas-Kerstjens',
    author_email='joserapa98@gmail.com',
    license='MIT',
    python_requires='>=3.8',
    install_requires=['torch>=1.9',
                      'opt_einsum>=3.0'],
    extras_require={
        'tests': ['pytest'],
        'docs': ['sphinx>=4.5',
                 'sphinx-book-theme==0.3.3',
                 'sphinx-copybutton',
                 'nbsphinx']
    },
    packages=[LIB_NAME, 'tensorkrowch.models'],
    ext_modules=[cpp_extension],
    cmdclass={'build_ext': BuildExtension},
)
