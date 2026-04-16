from src.core.grid import EulerianGrid


def test_anisotropic_spacing_2d_uses_cartesian_input_order():
    grid = EulerianGrid(shape=(5, 7), dx=(2.0, 3.0))
    assert grid.dx == 2.0
    assert grid.dy == 3.0
    assert grid.spacing == (3.0, 2.0)


def test_anisotropic_spacing_3d_uses_cartesian_input_order():
    grid = EulerianGrid(shape=(4, 5, 6), dx=(2.0, 3.0, 4.0))
    assert grid.dx == 2.0
    assert grid.dy == 3.0
    assert grid.dz == 4.0
    assert grid.spacing == (4.0, 3.0, 2.0)
