"""Tsufvml package definition and install configuration"""

# Copyright (c) 2018 Aubrey Barnard.  This is free software released
# under the MIT License.  See `LICENSE.txt` for details.

# References:
# https://packaging.python.org/
# https://setuptools.readthedocs.io/en/latest/setuptools.html
# https://pip.pypa.io/en/latest/
# https://www.python.org/dev/peps/pep-0440/


import setuptools

import tsufvml


# Get the description from the package documentation
_desc_paragraphs = tsufvml.__doc__.strip().split('\n\n')
_desc_short = _desc_paragraphs[0].replace('\n', ' ') # Needs to be one line
_desc_long = _desc_paragraphs[1]


# Define package attributes
setuptools.setup(

    # Basic characteristics
    name='tsufvml',
    version=tsufvml.__version__,
    url='https://github.com/afbarnard/tsufvml',
    license='MIT',
    author='Aubrey Barnard',
    #author_email='',

    # Description
    description=_desc_short,
    long_description=_desc_long,
    keywords=[
        'data science',
        'machine learning',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],

    # Requirements
    python_requires='~=3.4',
    #install_requires=[
        # Do not include scikit-learn as a prerequisite because it
        # installs well with `pip` only if NumPy and SciPy are already
        # installed.  Therefore attempting an install is unpredictable
        # and may lead to attempting to build everything from scratch,
        # which is definitely not desired.

        #'scikit-learn ~= 0.17, < 0.20',
    #],
    extras_require={
        'graphviz': ['pydot~=1.1'],
    },

    # API
    packages=setuptools.find_packages(),
    #entry_points={}, # for scripts

)
