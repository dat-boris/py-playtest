import inspect
from typing import Dict, Type, Sequence, Union, TypeVar, Generic
import numpy as np
from enum import IntEnum

import gym.spaces as spaces

from .components.core import Component


class Visibility(IntEnum):
    """Represent visibility - the higher it is, more visible it is"""

    ALL = 3
    SELF = 2
    NONE = 1

    @staticmethod
    def is_visible_to(
        comp_visibility: "Visibility", min_visibility: "Visibility"
    ) -> bool:
        return comp_visibility >= min_visibility


class SubState(Component):
    """Describe a particular instance of a state

    Can contain other instance of states
    """

    visibility: Dict[str, Visibility]

    def __init__(self, param=None):
        pass

    def reset(self):
        for name in self.visibility.keys():
            attr_val = getattr(self, name, None)
            assert attr_val is not None, f"Instance did not have value {name}"
            if isinstance(attr_val, list):
                for attr_val_in_list in attr_val:
                    attr_val_in_list.reset()
            else:
                attr_val.reset()

    def to_data(self, to_data_func_name="to_data"):
        return self._to_data_from_spec(
            Visibility.NONE, to_data_func_name=to_data_func_name
        )

    def to_visible_data(self, to_data_func_name="to_data"):
        return self._to_data_from_spec(
            Visibility.ALL, to_data_func_name=to_data_func_name
        )

    # TODO: Note that this is should be classmethod
    def get_observation_space(self) -> spaces.Space:  # type: ignore
        """Get visible observational space.

        Notice that we use the suffix `_full` for the naming, as this
        describe, since we want to "default" to use get_observation_space
        for components
        """
        return self.get_observation_space_visible()

    def get_observation_space_visible(self) -> spaces.Space:
        obs_dict = self._to_data_from_spec(
            Visibility.ALL, to_data_func_name="get_observation_space",
        )
        return spaces.Dict(obs_dict)

    def get_observation_space_full(self) -> spaces.Space:
        obs_dict = self._to_data_from_spec(
            Visibility.NONE, to_data_func_name="get_observation_space",
        )
        return spaces.Dict(obs_dict)

    def to_visible_numpy_data(self):
        return self._to_data_from_spec(
            Visibility.ALL, to_data_func_name="to_numpy_data",
        )

    def _to_data_from_spec(
        self, min_visibility: Visibility, to_data_func_name="to_data",
    ) -> Dict:
        """Convert data based on the given spec"""
        val_dict: Dict = {}
        for name, visibility in self.visibility.items():
            if not Visibility.is_visible_to(visibility, min_visibility):
                continue
            attr_val = getattr(self, name, None)
            # if isinstance(attr_val, list):
            #     val_dict[name] = []
            #     for attr_val_in_list in attr_val:
            #         assert isinstance(attr_val_in_list, data_class)
            #         # e.g. attr_val_in_list.to_data()
            #         to_data_func = getattr(attr_val_in_list, to_data_func_name)
            #         val_dict[name].append(to_data_func())
            if isinstance(attr_val, Component):
                # e.g. attr_va.to_data()
                to_data_func = getattr(attr_val, to_data_func_name)
                val_dict[name] = to_data_func()
            elif isinstance(attr_val, int):
                val_dict[name] = attr_val
            else:
                raise RuntimeError(
                    f"Received non component or int in {name}: {attr_val}"
                )
        return val_dict

    @classmethod
    def from_data(cls, data):
        instance = cls()
        for name, data_class in cls.__annotations__.items():
            assert name in data, "{} does not present in data.".format(name)
            if inspect.isclass(data_class) and issubclass(data_class, Component):
                attr_instance = data_class.from_data(data[name])
                assert isinstance(
                    attr_instance, Component
                ), f"{data_class} did not return component"
                setattr(instance, name, attr_instance)
        return instance


S = TypeVar("S", bound="SubState")


class FullState(SubState, Generic[S]):
    """Define a general state representation

    This also contains a set of player SubState, which we will need to handle
    gracefully.
    """

    player_state_class: Type[S]
    players: Sequence[S]
    current_player_id: int

    def __init__(self, param=None):
        """Initialize the players

        Params is only required, if we are not initializing from scratch
        """
        self.players = []
        if param is not None:
            for _ in range(param.number_of_players):
                self.players.append(self.player_state_class(param))

    @property
    def number_of_players(self) -> int:
        return len(self.players)

    def next_player(self) -> int:
        self.current_player_id = (self.current_player_id + 1) % self.number_of_players
        return self.current_player_id

    def reset(self):
        super().reset()
        for player_state in self.players:
            player_state.reset()

    def to_data(self):
        data_output = super(FullState, self).to_data()

        assert (
            "players" not in data_output
        ), "Cannot contain players for the default data output"

        data_output["players"] = []

        for player_state in self.players:
            data_output["players"].append(player_state.to_data())

        return data_output

    @classmethod
    def from_data(cls, data):
        instance = super().from_data(data)

        assert "players" in data, f"Given data should contain players key {data}"

        instance.players = []
        for player_data in data["players"]:
            player_state = cls.player_state_class.from_data(player_data)
            instance.players.append(player_state)

        return instance

    def get_player_state(self, player_id: int) -> S:
        assert isinstance(player_id, int)
        return self.players[player_id]

    def to_player_data(self, player_id: int, for_numpy=False) -> Dict:
        """Return data structure for numpy related setup

        :for_numpy: return if this is consumed by numpy.  This will fill
          the array of data accordingly
        """
        to_data_func_name = "to_data"
        if for_numpy:
            to_data_func_name = "to_data_for_numpy"
        all_data = self._to_data_from_spec(
            Visibility.SELF, to_data_func_name=to_data_func_name
        )
        all_data["self"] = {}
        all_data["others"] = []

        for pid, player_state in enumerate(self.players):
            if pid == player_id:
                all_data["self"] = player_state.to_data(
                    to_data_func_name=to_data_func_name
                )
            else:
                # Only add visible data
                all_data["others"].append(
                    player_state.to_visible_data(to_data_func_name=to_data_func_name)
                )

        return all_data

    @classmethod
    def get_observation_space(cls):
        raise NotImplementedError("Use get_observation_space_from_player instead.")

    def get_observation_space_from_player(self) -> spaces.Space:
        obs_dict = self._to_data_from_spec(
            Visibility.SELF, to_data_func_name="get_observation_space",
        )

        example_state = self.players[0]
        obs_dict["self"] = example_state.get_observation_space_full()

        number_of_players = len(self.players)
        obs_dict["others"] = spaces.Tuple(
            [example_state.get_observation_space_visible()] * (number_of_players - 1)
        )
        return spaces.Dict(obs_dict)
