import pytest

from .constant import Param
from .state import State

NUMBER_OF_PLAYERS = 2


@pytest.fixture
def state():
    param = Param(number_of_players=NUMBER_OF_PLAYERS)
    state = State(param)
    return state


def test_serialize(state):
    """Test that we can serialize the data
    """
    st_data = state.to_data()

    assert st_data, "Data is what we expected"
    assert len(st_data["deck"]) == 52
    assert len(st_data["discarded"]) == 0

    player_state = st_data["players"][0]
    assert player_state["bank"] == [10]
    assert player_state["bet"] == [0]

    new_state = State.from_data(st_data)

    st_data_new = new_state.to_data()
    assert len(st_data_new["deck"]) == 52

    assert st_data == st_data_new


def test_player_initialize(state):
    st_data = state.to_data()

    assert len(st_data["players"]) == NUMBER_OF_PLAYERS
