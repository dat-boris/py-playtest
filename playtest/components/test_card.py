import pytest
import numpy as np

import gym.spaces as spaces

from .card import Card, CardNumber, CardSuite


def test_card():
    """Testing counter as a simple conversion
    """
    with pytest.raises(TypeError):
        Card(value=(1, 3))

    c = Card(value=(CardNumber.T, CardSuite.S))
    c2 = Card(value=(CardNumber._9, CardSuite.D))

    data_representation = c.to_data()
    # TODO: note that should we represent as enum?
    assert data_representation == (10, 1)
    new_obj = Card.from_data(data_representation)
    assert new_obj.value == (10, 1)

    assert isinstance(c.observation_space, spaces.Box)
    # Note the box conversion convert this to an array
    assert (c.to_numpy_data() == np.array([10, 1])).all()

    # Comparison depends on numpy
    assert new_obj == c, "Comparison works with numpy"

    assert repr(c) == "T,S", "Str representation works"
    assert Card.from_str("T,S") == c
    assert repr(c2) == "_9,D", "Str representation works"
    assert Card.from_str("_9,D") == c2
