import enum
import pytest
import numpy as np

import gym.spaces as spaces

from playtest.components.resource import Resource


class FooBar(enum.IntEnum):
    F = 1
    B = 2


class FooBarResource(Resource[FooBar]):
    generic_resource = FooBar
    # Note this must map to the number of elements in FooBar
    value_type = (int, int)


def test_resources():
    """## Resources

    Resource is a generic class, that can specify different type of resource.


    Each of these resources are in order of the value, and there are upgrade cards
    that can upgrade these resources to the next level.

    See card_upgrade for detail.
    """
    r = FooBarResource([2, 3])
    assert r.to_data() == [2, 3], "Represent resources as array of integer"

    assert r.stack == {
        FooBar.F.name: 2,
        FooBar.B.name: 3,
    }

    r2 = FooBarResource.from_data([2, 3])
    assert r2 == r

    assert isinstance(r.get_observation_space(), spaces.Box)
    assert (r.to_numpy_data() == np.array([2, 3])).all()

    # TODO (boris): Not quiet human friendly!
    assert repr(r) == "FooBarResource(2,3)", "Simple string presentation"
    assert FooBarResource.from_str("2,3") == r


def test_resource_comparison():
    r = FooBarResource([2, 3])

    assert r.has_required(FooBarResource([2, 1])) is True
    assert r.has_required(FooBarResource([4, 5])) is False
    assert r.has_required(FooBarResource([4, 1])) is False

