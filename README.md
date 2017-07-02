Temporal Studies using Feature Vector Machine Learning
======================================================


Tsufvml ('tsuf vehmul) is software for conducting temporal studies
(e.g. case-control studies) using typical machine learning methods that
use feature vectors.


License
-------

This software is free, open source software.  It is released under the
MIT License, contained in the file `LICENSE.txt`.


Requirements
------------

* Python >= 3.4
* Scikit-Learn >= 0.17 (< 0.20)
* Fitamord >= 0.1


Installation
------------

1. Install Scikit-Learn.

   [Install Scikit-Learn](http://scikit-learn.org/stable/install.html)
   using your operating system's package manager or using
   [Miniconda](https://conda.io/docs/install/quick.html) or
   [Anaconda](https://www.continuum.io/anaconda-overview).  Unless your
   operating system is a Linux distribution where you have
   administrative privileges, Miniconda is probaby what you want because
   it provides self-contained environments that allow you to have
   specific Python versions with exactly the packages you need.

   Miniconda is the basic environment, while Anaconda comes with a large
   set of packages included.  Thus Miniconda is a much smaller download
   (about 35MB).  Both provide the Conda package manager.  In either
   case, use the Python 3 versions (Miniconda3, Anaconda3).  Note that
   Conda is released under the [BSD 3-Clause
   License](https://conda.io/docs/license.html).

   To install and update Miniconda, run the following commands.
   (Naturally, adjust these for your OS and architecture.  See
   https://conda.io/miniconda.html.)

       wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
       bash Miniconda3-latest-Linux-x86_64.sh
       conda --version
       conda update conda

   To install Scikit-Learn with Conda run the following commands.  Like
   Python, Conda uses environments to isolate installations of packages
   from one another, so it's best to create an environment for this
   software rather than use the default (root) environment.

       conda create --name tsufvml python=3 scikit-learn

   This is equivalent to creating an environment and then installing
   into it.

       conda create --name tsufvml
       source activate tsufvml
       conda install python=3 scikit-learn

2. Create a workspace and (optionally) a virtual environment.

   Create a workspace for Tsufvml by creating a directory for it and
   changing to it.  You can pick any directory you want instead of
   `~/workspaces/tsufvml`.

       mkdir -p ~/workspaces/tsufvml && cd ~/workspaces/tsufvml

   Next create a virtual environment and activate it.  This is optional
   but recommended.  A virtual environment is isolated from the system's
   Python environment and other virtual environments.  It allows you to
   install exactly the packages an application needs, even if they would
   otherwise conflict with system packages.

   Note that Python virtual environments live in a specific directory
   but Conda virtual environments are not associated with any particular
   directory.  Thus, Conda environments can be activated from any
   directory, but the directories for your workspace and Python
   environment will typically coincide.

   If you already created a Conda environment above, you don't need to
   create an additional environment, just activate it.  (Note that,
   here, `activate` is a program provided by Conda.)

       source activate tsufvml

   If you are not using Conda, proceed by creating and activating a
   Python virtual environment.  This assumes Scikit-Learn has been
   installed as a system package.

       python3 -m venv --system-site-packages ~/workspaces/tsufvml
       cd ~/workspaces/tsufvml
       source bin/activate

   Then upgrade Pip and Setuptools in your environment (optional but
   recommended).

   With Conda:

       conda update pip setuptools

   With Pip:

       python3 -m pip install --upgrade pip setuptools

3. Install Fitamord (data processing prerequisite).

   Using your workspace, follow the instructions at
   https://github.com/afbarnard/fitamord/blob/master/README.md to
   install Fitamord.  They boil down to running the following command.
   Note the version number.

       python3 -m pip install --editable git+https://github.com/afbarnard/fitamord.git@v0.1.0#egg=fitamord

3. Install Tsufvml.

   Using your workspace, run the following command:

       python3 -m pip install --editable git+https://github.com/afbarnard/tsufvml.git#egg=tsufvml

   If you want to install Tsufvml user-wide instead of just in your workspace, add the `--user` option:

       python3 -m pip install --user --editable git+https://github.com/afbarnard/tsufvml.git#egg=tsufvml


Usage
-----

First, tell Fitamord how to handle your data by generating and editing
its configuration.  See the usage instructions in the
[README](https://github.com/afbarnard/fitamord/blob/master/README.md).

Next, use Fitamord to generate a feature vector version of your data.
This may take several hours to run depending on the size of your data.

Finally, run Tsufvml to analyze your data.  (Only decision tress are
supported at the moment.)

    python3 <path>/<to>/tsufvml/tsufvml/run_sklearn_decision_trees.py <path>/<to>/feature_vector_data.svmlight > report.yaml

You can visualize the learned tree using `dot`.

    sed -n -e '/digraph/,/}/ p' report.yaml > tree.dot # Excerpt the tree definition (or do by hand)
    dot -Tpdf tree.dot > tree.pdf


-----

Copyright (c) 2017 Aubrey Barnard.  This is free software released under
the MIT License.  See `LICENSE.txt` for details.
