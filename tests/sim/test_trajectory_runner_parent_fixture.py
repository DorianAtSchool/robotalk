
import numpy as np
import pytest

from robocasa.utils.trajectory_runner import _countertop_support_parent


class _Fixture:
    def __init__(self, minimum, maximum):
        minimum = np.asarray(minimum, dtype=float)
        maximum = np.asarray(maximum, dtype=float)
        self.pos = (minimum + maximum) / 2.0
        self._points = [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]

    def get_ext_sites(self, *, all_points, relative):
        assert all_points and not relative
        return self._points


def test_countertop_parent_uses_surface_below_not_overhead_cabinet():
    coffee = _Fixture([5.75, -3.39, 0.92], [6.20, -3.26, 1.25])
    counter = _Fixture([5.20, -4.00, 0.40], [6.30, -2.80, 0.92])
    cabinet = _Fixture([5.70, -3.80, 1.30], [6.30, -2.80, 2.20])
    fixtures = {"coffee": coffee, "counter": counter, "cabinet": cabinet}
    fixture_types = {
        "coffee": "coffee_machine",
        "counter": "counter_non_dining",
        "cabinet": "cabinet_double_door",
    }

    assert _countertop_support_parent(
        "coffee", coffee, fixtures, fixture_types
    ) == "counter"


def test_countertop_parent_selects_larger_overlapping_counter_segment():
    toaster = _Fixture([0.90, 0.90, 0.90], [1.30, 1.20, 1.20])
    small_overlap = _Fixture([0.00, 0.00, 0.40], [0.95, 2.00, 0.90])
    large_overlap = _Fixture([0.95, 0.00, 0.40], [2.00, 2.00, 0.90])
    fixtures = {
        "toaster": toaster,
        "small": small_overlap,
        "large": large_overlap,
    }
    fixture_types = {
        "toaster": "toaster_oven",
        "small": "counter_non_dining",
        "large": "counter_non_dining",
    }

    assert _countertop_support_parent(
        "toaster", toaster, fixtures, fixture_types
    ) == "large"


def test_countertop_parent_fails_when_no_surface_is_below():
    appliance = _Fixture([0.0, 0.0, 1.0], [0.4, 0.4, 1.3])
    cabinet = _Fixture([0.0, 0.0, 1.4], [0.5, 0.5, 2.0])
    with pytest.raises(RuntimeError, match="Cannot resolve supporting surface"):
        _countertop_support_parent(
            "coffee",
            appliance,
            {"coffee": appliance, "cabinet": cabinet},
            {"coffee": "coffee_machine", "cabinet": "cabinet_double_door"},
        )
