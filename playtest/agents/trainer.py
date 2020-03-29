from typing import Sequence, List
import numpy as np
import pdb, traceback, sys

# This is dependent on work from
# https://github.com/keras-rl/keras-rl/compare/master...dat-boris:add-multi-agent
# TODO: port this to rllib, which seems to already have good multi-agent support
from rl.agents.multi import MultiAgent
from keras.optimizers import Adam


from playtest.env import GameWrapperEnvironment
from playtest.agents.base import BaseAgent


DEFAULT_NB_STEPS = 500000


def train_agents(
    env: GameWrapperEnvironment,
    agents: Sequence[BaseAgent],
    save_filenames: List[str],
    nb_steps=DEFAULT_NB_STEPS,
    is_pdb=False,
):
    assert save_filenames, "Must save at last one agent"
    assert all(
        [s.endswith(".h5f") for s in save_filenames]
    ), f"{save_filenames} should all end with h5f extension"
    np.random.seed(123)
    env.seed(123)

    multi_agent = MultiAgent(agents)
    assert all(
        [getattr(a, "target_model", None) for a in agents]
    ), "Must have compiled the model"
    multi_agent.compile(Adam(lr=1e-3), metrics=["mae"])

    try:
        multi_agent.fit(
            env,
            nb_steps=nb_steps,
            # visualize=True, verbose=2
        )
    except Exception as ex:
        if is_pdb:
            _, _, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        raise ex

    for i, file_name in enumerate(save_filenames):
        model = agents[i]
        print(f"Saving model {model} filename: {file_name}")
        model.save_weights(file_name, overwrite=True)
