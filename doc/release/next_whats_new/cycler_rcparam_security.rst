``axes.prop_cycle`` rcParam must be literal
--------------------------------------------

The ``axes.prop_cycle`` rcParam is now parsed safely without ``eval()``. Only
literal ``cycler()`` calls combined with ``+`` and ``*`` are allowed. All
previously valid cycler strings continue to work, for example:

.. code-block:: none

   axes.prop_cycle : cycler('color', ['r', 'g', 'b']) + cycler('linewidth', [1, 2, 3])
