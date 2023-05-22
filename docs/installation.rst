Installation
============

To install the package, run the following command:

::

    $ pip install tensorkrowch

You can also install directly from GitHub with:

::

    $ pip install git+https://github.com/joserapa98/tensorkrowch.git@master

or download the repository on your computer and run 

::

    $ pip install .

in the repository folder.


.. warning::

    Since **TensorKrowch** has a `PyTorch C++ Extension
    <https://pytorch.org/tutorials/advanced/cpp_extension.html>`_, it has to be 
    built from source, so make sure you have installed on your system a C++
    compiler compatible with C++14.

Tests are written outside the Python module, therefore they are not installed
together with the package. To test the installation, clone the repository and
run, in a Unix terminal

::
    
    $ python -m pytest

inside the repository folder.
