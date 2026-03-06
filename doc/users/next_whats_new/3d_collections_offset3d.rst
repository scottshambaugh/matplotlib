3D Collections have ``set_offsets3d`` and ``get_offsets3d`` methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All 3D Collections (``Patch3DCollection``, ``Path3DCollection``,
``Poly3DCollection``) now have ``set_offsets3d`` and ``get_offsets3d`` methods
which allow you to set and get the offsets of the collection in data
coordinates.

For ``Patch3DCollection`` and ``Path3DCollection`` (e.g. from `~.Axes3D.scatter`),
this sets the position of each element. For ``Poly3DCollection``, the offsets
translate the polygon faces in 3D space.
