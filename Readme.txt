

Other dependencies:

For Windows:

    C++ compiler
    https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2017

    MATLAB API for Python
    https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html
    cd matlabroot\extern\engines\python
    python setup.py build --build-base='builddir' install --user

For Linux:

    MATLAB API for Python
    https://www.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html
    cd matlabroot\extern\engines\python
    python setup.py build --build-base='builddir' install --user


To install:

    cd interaction3
    python setup.py build_ext --inplace