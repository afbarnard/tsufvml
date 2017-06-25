"""Tsufvml package definition and install configuration"""

# Copyright (c) 2017 Aubrey Barnard.  This is free software released
# under the MIT License.  See `LICENSE.txt` for details.


import setuptools

import tsufvml.version


# Define package attributes
setuptools.setup(

    # Basic characteristics
    name='tsufvml',
    version=tsufvml.version.__version__,
    url='https://github.com/afbarnard/tsufvml',
    license='MIT',
    author='Aubrey Barnard',
    #author_email='',

    # Description
    description=(
        'Tsufvml is software for conducting temporal studies (e.g. '
        'case-control studies) using typical machine learning methods '
        'that use feature vectors.'
        ),
    #long_description='',
    keywords=[
        'relational data',
        'data preparation',
        'data modeling',
        'feature functions',
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
    python_requires='>=3.4',
    #install_requires=[],

    # API
    packages=setuptools.find_packages(),
    #entry_points={}, # for scripts

    )
