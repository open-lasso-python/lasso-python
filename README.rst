
|LASSO| Python Library
======================

|test-main| |test-dev|

.. |test-main| image:: https://github.com/open-lasso-python/lasso-python/actions/workflows/ci-cd.yml/badge.svg?branch=main
   :target: https://github.com/open-lasso-python/lasso-python/actions/workflows/test-runner.yml

.. |test-dev| image:: https://github.com/open-lasso-python/lasso-python/actions/workflows/ci-cd.yml/badge.svg?branch=develop
   :target: https://github.com/open-lasso-python/lasso-python/actions/workflows/test-runner.yml

This python library is designed for general purpose usage in the field of
Computer Aided Engineering (CAE).
It's name originates from the original initiator and donator of the project
`LASSO GmbH`_.
The library is now maintained by an open-source community.

Module Overview:
 - `lasso.dyna`_
 - `lasso.dimred`_
 - `lasso.femzip`_
 - `lasso.diffcrash`_

For further infos please read the Documentation:

    |DOCS| `Documentation`_

.. _LASSO GmbH: https://www.lasso.de/en
.. _Documentation: https://open-lasso-python.github.io/lasso-python/
.. _lasso.dyna: https://open-lasso-python.github.io/lasso-python/dyna/
.. _lasso.dimred: https://open-lasso-python.github.io/lasso-python/dimred/
.. _lasso.femzip: https://open-lasso-python.github.io/lasso-python/femzip/
.. _lasso.diffcrash: https://open-lasso-python.github.io/lasso-python/diffcrash/


Installation
------------

..  code-block:: bash

    python -m pip install lasso-python


Community
---------

Join our open-source community on: 

    |DISCORD| `Discord`_
 
.. _Discord:  https://discord.gg/jYUgTsEWtN

.. |LASSO| image:: ./docs/lasso-logo.png
    :target: https://open-lasso-python.github.io/lasso-python/build/html/index.html
.. |DOCS| image:: ./docs/icon-home.png 
    :target: https://open-lasso-python.github.io/lasso-python/build/html/index.html
.. |DISCORD| image:: ./docs/icon-discord.png
    :target: https://discord.gg/GeHu79b


Development
-----------

For development install `poetry`_ and `task`_:

..  code-block:: bash

    python -m pip install poetry
    sh -c "$(curl --location https://taskfile.dev/install.sh)" \
        -- -d -b ~/.local/bin

Then by simply running the command ``task`` you can find a variety of available
commands such as ``task setup`` to install all dependencies or ``task test`` to
run the test suite.
Happy Coding 🥳🎉

.. _poetry: https://python-poetry.org/
.. _task: https://taskfile.dev/
