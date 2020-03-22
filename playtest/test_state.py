"""This define a set of state related operation
"""
import pytest

import gym.spaces as spaces

from .constant import Param
from .components.card import BasicDeck
from .state import SubState, FullState, Visibility


class MockPlayerState(SubState):

    visibility = {
        "hand": Visibility.SELF,
        "open_hand": Visibility.ALL,
    }

    hand: BasicDeck
    open_hand: BasicDeck

    def __init__(self, param=None):
        self.hand = BasicDeck([])
        self.open_hand = BasicDeck([])


class MockState(FullState):

    player_state_class = MockPlayerState

    visibility = {
        "deck": Visibility.NONE,
        "discarded": Visibility.ALL,
    }

    deck: BasicDeck
    discarded: BasicDeck

    def __init__(self, param=None):
        self.deck = BasicDeck(all_cards=True, shuffle=True)
        self.discarded = BasicDeck([])
        super().__init__(param=param)


@pytest.fixture
def state():
    param = Param(number_of_players=2)
    state = MockState(param)
    return state


def test_serialize(state):
    """Test that we can serialize the data
    """
    assert len(state.players) == 2
    st_data = state.to_data()

    assert st_data, "Expect that we would have some data!"
    assert len(st_data["deck"]) == 52
    assert len(st_data["discarded"]) == 0
    # Render player subset properly
    assert len(st_data["players"]) == 2
    assert len(st_data["players"][0]["hand"]) == 0

    new_state = MockState.from_data(st_data)
    assert new_state.__class__ == MockState
    st_data_new = new_state.to_data()

    assert st_data == st_data_new


def test_visible_data(state):
    """Getting only the visible data"""
    st_data = state.to_player_data(0)

    assert st_data, "Expect that we would have some data!"
    assert "deck" not in st_data, "We should not see the deck"
    assert len(st_data["discarded"]) == 0, "We should see discarded"

    # Should see all data of the player self
    assert len(st_data["self"]["hand"]) == 0

    # Should not see other player's data
    other_hand = st_data["others"][0]
    assert "hand" not in other_hand
    assert len(other_hand["open_hand"]) == 0


def test_to_numpy(state):
    st_data = state.to_player_numpy_data(0)

    assert st_data, "Expect that we would have some data!"
    assert "deck" not in st_data, "We should not see the deck"
    assert len(st_data["discarded"]) == 52, "We should see discarded"

    # Should see all data of the player self, max size
    assert st_data["self"]["hand"].tolist() == [0] * 52

    # Should not see other player's data
    other_hand = st_data["others"][0]
    assert "hand" not in other_hand
    assert other_hand["open_hand"].tolist() == [0] * 52


def test_observational_space(state):
    st_data = state.observation_space

    assert st_data, "Expect that we would have some data!"
    assert isinstance(st_data, spaces.Dict)
    assert "deck" not in st_data, "We should not see the deck"
    assert isinstance(st_data["discarded"], spaces.Box)
    assert st_data["discarded"].shape == (52,)

    # Should see all data of the player self, max size
    assert isinstance(st_data["self"], spaces.Dict)
    assert isinstance(st_data["self"], spaces.Dict)
    assert isinstance(st_data["self"]["hand"], spaces.Box)

    # Should not see other player's data
    assert isinstance(st_data["others"], spaces.Tuple)
    assert len(st_data["others"]) == 1
    other_hand = st_data["others"][0]
    assert "hand" not in other_hand
    assert isinstance(other_hand["open_hand"], spaces.Box)
