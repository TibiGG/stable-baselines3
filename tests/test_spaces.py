import gym
import numpy as np
import pytest

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.td3.matd3 import MATD3


class DummyMultiDiscreteSpace(gym.Env):
    def __init__(self, nvec):
        super().__init__()
        self.observation_space = gym.spaces.MultiDiscrete(nvec)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, {}


class DummyMultiBinary(gym.Env):
    def __init__(self, n):
        super().__init__()
        self.observation_space = gym.spaces.MultiBinary(n)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, {}
        # reward = tuple(self._rewards[index] for _ in range(self._n_agents))
        reward = self._rewards[index]
        return obs, reward, done, {}


class DummyDict(gym.Env):
    def __init__(self):
        super().__init__()
        space = gym.spaces.Box(1, 5, (1,))
        self.observation_space = gym.spaces.Dict({"observation": space, "achieved_goal": space, "desired_goal": space})
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, {}


class DummyTupleDict(gym.Env):
    def __init__(self, n_agents):
        super().__init__()
        self.n_agents = n_agents
        space = gym.spaces.Box(1, 5, (1,))
        self.observation_space = gym.spaces.Tuple(tuple(gym.spaces.Dict({"observation": space, "achieved_goal": space, "desired_goal": space})
                                                    for _ in range(self.n_agents)))
        self.action_space = gym.spaces.Tuple(
            tuple(gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
                  for _ in range(self.n_agents)))

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, {}


@pytest.mark.parametrize("model_class", [SAC, TD3, DQN])
@pytest.mark.parametrize("env", [DummyMultiDiscreteSpace([4, 3]), DummyMultiBinary(8), DummyDict()])
def test_identity_spaces(model_class, env):
    """
    Additional tests for DQ/SAC/TD3 to check observation space support
    for MultiDiscrete and MultiBinary.
    """
    # DQN only support discrete actions
    if model_class == DQN:
        env.action_space = gym.spaces.Discrete(4)

    # Dict and TupleDict envs only support MultiInputPolicy
    if isinstance(env, DummyTupleDict):
        policy = "MultiAgentMultiInputPolicy"
    elif isinstance(env, DummyDict):
        policy = "MultiInputPolicy"
    else:
        policy = "MlpPolicy"
    is_marl = isinstance(env, DummyTupleDict)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

    # TODO: consider bringing is_marl as parameter for all off-policy model classes
    if model_class != DQN:
        model = model_class(policy, env, gamma=0.5, seed=1, policy_kwargs=dict(net_arch=[64]),
                            is_marl=is_marl)
    else:
        model = model_class(policy, env, gamma=0.5, seed=1, policy_kwargs=dict(net_arch=[64]))

    model.learn(total_timesteps=500)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)


@pytest.mark.parametrize("model_class", [MATD3])
@pytest.mark.parametrize("env", [DummyTupleDict(3)])
def test_multi_agent_identity_spaces(model_class, env):
    """
    Additional tests for MATD3 to check observation space support
    for MultiAgent.
    """

    # Dict and TupleDict envs only support MultiInputPolicy
    policy = "MultiAgentMultiInputPolicy"
    is_marl = True

    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

    # TODO: consider bringing is_marl as parameter for all off-policy model classes
    model = model_class(policy, env, gamma=0.5, seed=1, policy_kwargs=dict(net_arch=[64]),
                        is_marl=is_marl)

    model.learn(total_timesteps=500)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)


@pytest.mark.parametrize("model_class", [A2C, DDPG, DQN, PPO, SAC, TD3])
@pytest.mark.parametrize("env", ["Pendulum-v1", "CartPole-v1"])
def test_action_spaces(model_class, env):
    if model_class in [SAC, DDPG, TD3]:
        supported_action_space = env == "Pendulum-v1"
    elif model_class == DQN:
        supported_action_space = env == "CartPole-v1"
    elif model_class in [A2C, PPO]:
        supported_action_space = True

    if supported_action_space:
        model_class("MlpPolicy", env)
    else:
        with pytest.raises(AssertionError):
            model_class("MlpPolicy", env)


@pytest.mark.parametrize("model_class", [A2C, PPO, DQN])
@pytest.mark.parametrize("env", ["Taxi-v3"])
def test_discrete_obs_space(model_class, env):
    env = make_vec_env(env, n_envs=2, seed=0)
    kwargs = {}
    if model_class == DQN:
        kwargs = dict(buffer_size=1000, learning_starts=100)
    else:
        kwargs = dict(n_steps=256)
    model_class("MlpPolicy", env, **kwargs).learn(256)
