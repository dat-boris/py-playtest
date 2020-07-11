import pytest

import numpy as np
import gym.spaces as spaces

from .counter import Counter


def test_counter():
    """Testing counter as a simple conversion
    """
    with pytest.raises(TypeError):
        c = Counter(value=3)

    c = Counter(value=[3])

    data_representation = c.to_data()
    assert data_representation == [3]
    new_obj = Counter.from_data(data_representation)
    assert new_obj.value == [3]

    assert isinstance(c.get_observation_space(), spaces.Box)
    # Note the box conversion convert this to an array
    assert c.to_data_for_numpy() == np.array([3])

    # Comparison depends on numpy
    assert new_obj == c, "Comparison works with numpy"

    assert repr(c) == "Counter(3)", "Str representation works"
    assert c.from_str("3") == c
