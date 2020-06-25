import abc
import enum
from collections import Counter
import numpy as np
from typing import Dict, List, TypeVar, Generic, Type

import gym.spaces as spaces

from playtest.components import Component


# This represent resources abstractly
R = TypeVar("R", bound=enum.IntEnum)


class Resource(Component, Generic[R]):
    """Constants for various types of resource
    """

    generic_resource: Type[R]

    value: List[int]

    @classmethod
    def get_max_amount(cls) -> int:
        """Return maximum amount for each resource
        """
        return 0xFF

    @property
    def stack(self) -> Dict[str, int]:
        """Convert value of resource into a structure
        """
        return {r.name: self.value[i] for i, r in enumerate(self.generic_resource)}

    @classmethod
    def upgrade_char(cls, s: R) -> R:
        return cls.generic_resource(s.value + 1)

    def __len__(self) -> int:
        """Return number of resources"""
        return sum(self.value)

    @classmethod
    def get_observation_space(cls):
        type_of_resource = len(cls.generic_resource)
        return spaces.Box(
            low=0, high=cls.get_max_amount(), shape=(type_of_resource,), dtype=np.uint8,
        )

    def has_required(self, required: "Resource") -> bool:
        my = self.stack
        theirs = required.stack
        assert set(my.keys()) == set(
            [s.name for s in self.generic_resource]
        ), f"Not all resources are there: {my.keys()} vs {set(self.generic_resource)}"
        return all([theirs.get(res, 0) <= my_count for res, my_count in my.items()])

    # TODO: fix these
    # def sub_resource(self, required: "Resource"):
    #     my = self.stack
    #     theirs = required.stack
    #     assert sorted(my.keys()) == self.get_all_resources()
    #     self.value = self.struct_to_value(
    #         {res: my_count - theirs.get(res, 0) for res, my_count in my.items()}
    #     )

    # def sub_with_remainder(self, required: "Resource") -> "Resource":
    #     my = self.stack
    #     theirs = required.stack
    #     assert sorted(my.keys()) == self.get_all_resources()
    #     result = {res: my_count - theirs.get(res, 0) for res, my_count in my.items()}
    #     new_my_value = {}
    #     remainder = {}
    #     for k, v in result.items():
    #         if result[k] < 0:
    #             remainder[k] = -v
    #             new_my_value[k] = 0
    #         else:
    #             new_my_value[k] = v
    #     self.value = self.struct_to_value(new_my_value)
    #     return Resource(self.struct_to_value(remainder))

    # def add_resource(self, required: "Resource"):
    #     my = self.stack
    #     theirs = required.stack
    #     assert sorted(my.keys()) == self.get_all_resources()
    #     self.value = self.struct_to_value(
    #         {res: my_count + theirs.get(res, 0) for res, my_count in my.items()}
    #     )

    # def pop_lowest(self: R, amount) -> R:
    #     """Pop number of cheapest resources"""
    #     assert len(self) >= amount, f"Must have {amount} resources"
    #     data = self.stack
    #     popped_data: Dict[str, int] = {}
    #     for c in self.all_resources:
    #         if data[c] > 0 and sum(popped_data.values()) <= amount:
    #             to_discard = min(amount, data[c])
    #             data[c] -= to_discard
    #             popped_data[c] = to_discard
    #             amount -= to_discard
    #     self.value = self.struct_to_value(data)
    #     return self.__class__(self.struct_to_value(popped_data))

