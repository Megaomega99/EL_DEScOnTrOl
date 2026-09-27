import matplotlib as mpl

from cartpole_lab import plotting


def test_apply_style_sets_the_shared_surface_and_ink():
    with mpl.rc_context():
        plotting.apply_style()
        assert mpl.rcParams["figure.facecolor"] == plotting.SURFACE
        assert mpl.rcParams["axes.labelcolor"] == plotting.TEXT_SECONDARY


def test_each_method_keeps_its_own_color():
    colors = list(plotting.METHOD_COLORS.values())
    assert len(set(colors)) == len(colors) == 8
    assert colors == list(plotting.SERIES)  # los métodos ocupan los slots en orden, sin ciclos


def test_categorical_order_is_fixed_and_distinct():
    assert len(set(plotting.SERIES)) == len(plotting.SERIES) == 8
    assert plotting.SERIES[0] == "#2a78d6"  # slot 1 = serie principal en todas las figuras
