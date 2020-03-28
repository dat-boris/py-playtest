"""A simple keras DQN agent

This creates a simple Keras DQN aganet.
"""

# for training related models
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.core import Agent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from gym import Env
from gym.spaces import flatdim

from playtest.env import GameWrapperEnvironment
from playtest.agents.base import BaseAgent


class KerasDQNAgent(BaseAgent, DQNAgent):

    env: GameWrapperEnvironment

    def __init__(self, env: GameWrapperEnvironment):
        """Build a simple DQN model"""
        BaseAgent.__init__(self, env)
        nb_actions: int = flatdim(env.action_space)

        model = Sequential()
        model.add(Flatten(input_shape=(1, flatdim(env.observation_space))))
        model.add(Dense(16))
        model.add(Activation("relu"))
        model.add(Dense(16))
        model.add(Activation("relu"))
        model.add(Dense(16))
        model.add(Activation("relu"))
        model.add(Dense(nb_actions))
        model.add(Activation("linear"))
        print(model.summary())

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=50000, window_length=1)
        policy = BoltzmannQPolicy()
        DQNAgent.__init__(
            self,
            model=model,
            nb_actions=nb_actions,
            memory=memory,
            nb_steps_warmup=10,
            target_model_update=1e-2,
            policy=policy,
        )

        # Ensure we compile an optimizer for target model
        self.compile(Adam(lr=1e-3), metrics=["mae"])
