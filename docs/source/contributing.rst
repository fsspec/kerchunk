.. _contributing:

************************
Contributing to kerchunk
************************

.. note::

  Large parts of this document are based on the
  `Xarray Contributing Guide <http://docs.xarray.dev/en/stable/contributing.html>`_
  , the `Pandas Contributing Guide <http://pandas.pydata.org/pandas-docs/stable/contributing.html>`_
  and the `xbatcher Contributing Guide <https://xbatcher.readthedocs.io/en/latest/contributing.html>`_.

.. warning::
    kerchunk is actively in development and breaking changes may be introduced at any point.
    In its current state, it is possible for experienced users to build kerchunk indices of datasets,
    which work today and will continue to work. End users of these reference sets will not be exposed
    to changes in kerchunk's code and usually don't even need to install kerchunk.


Bug reports and feature requests
================================

Bug reports or feature requests to the kerchunk project can be submitted by opening up an `Issue in the repository <https://github.com/fsspec/kerchunk/issues>`_.


Contributing code
==================

This project uses git for version control and github for issue tracking. If you need instructions on how to setup git, they can be found on `GitHub <https://help.github.com/set-up-git-redirect>`_.


.. _contributing.forking:

Creating a Fork
---------------

Once you have your git credentials setup, the next step is to create a fork of the project to work off of.
To fork the repo, navigate to the `kerchunk repository on github <https://github.com/fsspec/kerchunk>`_ and click the *Fork* button in the top right of the page.
This will create a fork of the kerchunk project in your own repository.

Next, you will want to clone this forked version of the repository onto the machine you are working on.
In your terminal/command prompt run:

.. code-block:: sh


    git clone git@github.com:<yourusername>/kerchunk.git
    cd kerchunk
    git remote add upstream git@github.com:fsspec/kerchunk.git


This will create a directory from your fork of the repository on your local machine and connect it to the main repository.


.. _contributing.dev_env:

Creating a development environment
----------------------------------

To test your code changes, you will need to build *kerchunk* from source, which
requires a Python environment.

.. _contributiong.dev_python:

Creating a Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To ensure that the python environment that you are using is the same one that everyone else is using,
it is necessary to create a virtual environment.
This will create an isolated development environment where you can install the kerchunk python dependencies.

- Install either `Anaconda <https://www.anaconda.com/download/>`_ or `miniconda
  <https://conda.io/miniconda.html>`_
- Make sure your conda is up to date (``conda update conda``)
- Make sure that you have :ref:`cloned the repository <contributing.forking>`
- ``cd`` to the *kerchunk* source directory


.. tip::
    If your conda build solving times are taking a long time, you can try `mamba <https://mamba.readthedocs.io/en/latest/installation.html#installation>`_,
    which is a mirror of conda written in c++`.

First we'll create and activate the build environment:

.. code-block:: sh

    conda env create --name kerchunk --file ci/environment-py3<*>.yml
    conda activate kerchunk


Now that you have the correct dependencies installed in your environment,
you should be able to install your development version of kerchunk locally.
In the projects home directory run:

.. code-block:: sh

    pip install -e .

To test that the installation was successful run:

.. code-block:: sh

   $ python  # start an interpreter
   >>> import kerchunk
   >>> kerchunk.__version__


To view your environments

.. code-block:: sh

      conda info --envs

To return to your base environment

.. code-block:: sh

      conda deactivate

See the full conda docs `here <http://conda.pydata.org/docs>`_.

Setting up pre-commit
~~~~~~~~~~~~~~~~~~~~~

We use `pre-commit <https://pre-commit.com/>`_ to manage code linting and style.
To set up pre-commit after activating your conda environment, run:

.. code-block:: sh

    pre-commit install

Now pre-commit will run whenever you create a git commit in the repository.
You may need to edit files that pre-commit has issues with and re-add them to the commit.

Creating a branch
-----------------


You want your ``main`` branch to reflect only production-ready code, so create a
feature branch before making your changes. For example

.. code-block:: sh

    git branch shiny-new-feature
    git checkout shiny-new-feature

The above can be simplified to

.. code-block:: sh

    git checkout -b shiny-new-feature

This changes your working directory to the shiny-new-feature branch.  Keep any
changes in this branch specific to one bug or feature so it is clear
what the branch brings to *kerchunk*. You can have many "shiny-new-features"
and switch in between them using the ``git checkout`` command.

To update this branch, you need to retrieve the changes from the ``main`` branch

.. code-block:: sh

    git fetch upstream
    git merge upstream/main

This will combine your commits with the latest *kerchunk* git ``main``.  If this
leads to merge conflicts, you must resolve these before submitting your pull
request.  If you have uncommitted changes, you will need to ``git stash`` them
prior to updating.  This will effectively store your changes, which can be
reapplied after updating.

Running the test suite
----------------------

*kerchunk* uses the `pytest <https://docs.pytest.org/en/latest/contents.html>`_
framework for testing. You can run the test suite using:

.. code-block:: sh

    pytest kerchunk

Ideally any new feature added should have test coverage.


Contributing documentation
==========================

Documentation improvements are appreciated. The documentation is contained within the ``docs`` directory of the project.
It is written in ``ReStructured Text (.rst)``, which is similar to markdown, but features more functionality.
These ReStructured text files are built into ``html`` using the python `sphinx (https://www.sphinx-doc.org/en/master/)_` package.

You can create a virtual environment by running:

.. code-block:: sh

    conda create --name kerchunk-docs python=3.8
    conda activate kerchunk-docs
    python -m pip install -r docs/requirements.txt

Once you make changes to the docs, you can build them with:

.. code-block:: sh

    cd docs
    make html

Contributing changes
====================

Once you feel good about your changes you can see them by typing:

.. code-block:: sh

    git status

If you have created a new file, it is not being tracked by git. Add it by typing:

.. code-block:: sh

    git add path/to/file-to-be-added.py



Now you can commit your changes in your local repository.

.. code-block:: sh

    git commit -m "<commit message>"

When you want your changes to appear publicly on your GitHub page, push your
commits to a branch off your fork.

.. code-block:: sh

    git push origin shiny-new-feature

Here ``origin`` is the default name given to your remote repository on GitHub.
You can see the remote repositories.

.. code-block:: sh

    git remote -v

If you navigate to your branch on GitHub, you should see a banner to submit a pull
request to the *kerchunk* repository.

.. raw:: html

    <script data-goatcounter="https://kerchunk.goatcounter.com/count"
            async src="//gc.zgo.at/count.js"></script>
