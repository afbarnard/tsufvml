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

* [Python](https://www.python.org/) ~= 3.4
* [Scikit-Learn](http://scikit-learn.org/) >= 0.18
* [Fitamord](https://github.com/afbarnard/fitamord) ~= 0.1 (optional if
  you already have data in SVMLight format)
* `dot` from [Graphviz](http://www.graphviz.org/) if you want to
  visualize the model trees
  * [`pydot`]( https://pypi.org/project/pydot/) if you want to generate
    a PDF of the model tree ([`graphviz`](
    https://pypi.org/project/graphviz/) also works)


Installation
------------

This is a step-by-step guide to installing Tsufvml, but it is helpful to
understand the material in the [tutorial on installing Python packages](
https://packaging.python.org/tutorials/installing-packages/).

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
   Python, Conda uses environments to isolate installations (sets of
   packages) from one another, so it's best to create an environment for
   this software rather than use the default (root) environment.

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

3. Install [Fitamord](https://github.com/afbarnard/fitamord) (optional
   data processing prerequisite).

   Use Fitamord if you need to generate feature vector data from
   relational data.  If you already have data in [SVMLight
   format](http://svmlight.joachims.org/) and you just want to analyze
   it, Fitamord is unnecessary.

   To install Fitamord, use your workspace to follow the [install
   instructions](https://github.com/afbarnard/fitamord#download-install).
   They boil down to running the following command.

       python3 -m pip install [--user] https://github.com/afbarnard/fitamord/archive/master.zip#egg=fitamord https://github.com/afbarnard/barnapy/archive/master.zip#egg=barnapy https://github.com/afbarnard/esal/archive/master.zip#egg=esal

4. Install Tsufvml.

   Using your workspace, run the following command:

       python3 -m pip install https://github.com/afbarnard/tsufvml/archive/master.zip#egg=tsufvml[graphviz]

   The trailing "[graphviz]" tells `pip` to also install Graphviz
   functionality.  This is optional but allows Tsufvml to generate a PDF
   of the learned decision tree (using the `pydot` Python package).

   If you want to install Tsufvml user-wide instead of just in your
   workspace, add the `--user` option:

       python3 -m pip install --user https://github.com/afbarnard/tsufvml/archive/master.zip#egg=tsufvml[graphviz]

   Note that `pydot` operates by calling out to the native Graphviz
   software, so you must have Graphviz installed on your system (at
   least the `dot` executable -- `dot.exe` on Windows).  To do that,
   follow these instructions on [how to install Graphviz](
   http://www.graphviz.org/download/).  You may need to also set your
   `PATH` so that the system can find `dot`.


Usage
-----

First, tell Fitamord how to handle your data by generating and editing
its configuration.  See the [usage
instructions](https://github.com/afbarnard/fitamord#usage).

Next, use Fitamord to generate a feature vector version of your data.
This may take several hours to run depending on the size of your data.

Finally, run Tsufvml to analyze your data.  (Only decision trees are
supported at the moment.)

    tsufvml_decision_tree <path>/<to>/feature_vector_data.svmlight > report.yaml

Note that, depending on where Tsufvml was installed and the contents of
your `PATH`, you may need to specify the path to
`tsufvml_decision_tree` to invoke it, i.e.,
`<prefix>/bin/tsufvml_decision_tree`.  For system-wide installs, the
prefix is usually `/usr` or `/usr/local`, for user-wide it is
`~/.local`, and for workspaces it is `.` (the current directory).
Activating a workspace sets up your `PATH`, so you shouldn't have to
worry about this when using a workspace.

If you have either of the `pydot` or `graphviz` Python packages
installed and a working `dot` executable, then Tsufvml can generate a
PDF of the learned decision tree (using the `--pdf` option).  Otherwise
you can generate the PDF yourself using `sed` and `dot`:

    sed -n -e '/digraph/,/}/ p' report.yaml > tree.dot # Excerpt the tree definition (or do by hand)
    dot -Tpdf tree.dot > tree.pdf


### Improving Reporting ###

In addition to rendering a decision tree as a PDF, you may want to
improve the comprehensibility of the output of Tsufvml by translating
internal feature IDs to meaningful names and descriptions.  There are
command line arguments that you can specify when you invoke
`tsufvml_decision_tree` that will do this for you: a feature table and a
concept table.  Specifying a feature table enables translating the
internal feature ID to a feature name from the table.  If you are
working with OMOP CDM data, then the feature name is probably not enough
because it only contains a concept ID.  In that case, specifying a
concept table enables translating the concept ID to a concept
description, which appears in a legend in the report.

For more help on the command line arguments, run `tsufvml_decision_tree
--help`.


### Excluding Features ###

If you run Tsufvml directly on the output of Fitamord you will get
perfect classification results.  This is because Fitamord includes the
ID and label of each example in the SVMLight data (for provenance,
debugging, etc.).  To avoid this trivial classification behavior,
specify a feature table with only the features you want to include in
the classification.  (When given a feature table, Tsufvml will include
only those features that are listed in the table.)  There are two ways
to do this: (1) comment out unwanted features by inserting a "#" at the
beginning of its line, or (2) copy the table of features generated by
Fitamord and delete the lines containing features you want to exclude.
Once you have limited your features, run Tsufvml while specifying your
new table of features.  Different sets of features can be kept in
different files to ease comparing various models; the data never needs
to change.


-----

Copyright (c) 2018 Aubrey Barnard.  This is free software released under
the MIT License.  See `LICENSE.txt` for details.
