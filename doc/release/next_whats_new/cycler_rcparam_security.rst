``axes.prop_cycle`` rcParam must be literal
--------------------------------------------

The ``axes.prop_cycle`` rcParam is now parsed safely without ``eval()``. Only
literal ``cycler()`` and ``concat()`` calls combined with ``+``, ``*``, and
slicing are allowed. All previously valid cycler strings continue to work,
for example:

.. code-block:: none

   axes.prop_cycle : cycler('color', ['r', 'g', 'b']) + cycler('linewidth', [1, 2, 3])
   axes.prop_cycle : 2 * cycler('color', 'rgb')
   axes.prop_cycle : concat(cycler('color', 'rgb'), cycler('color', 'cmk'))
   axes.prop_cycle : cycler('color', 'rgbcmk')[:3]
