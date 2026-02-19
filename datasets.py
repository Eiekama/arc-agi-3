import arc_agi
from arcengine import GameAction, FrameDataRaw, GameState, SimpleAction, ComplexAction
import gymnasium as gym
import numpy as np
from enum import Enum

class Environment:
    class Benchmark(Enum):
        ARC = "arc"
        ATARI = "atari"

    def __init__(
        self,
        benchmark: str,
        env: arc_agi.EnvironmentWrapper | gym.Env
    ):
        if (benchmark not in self.Benchmark):
            raise ValueError(f"[Environment] Unsupported benchmark: {benchmark}")
        
        self._env = env
        self.benchmark = self.Benchmark(benchmark)

        # Indices 0-7 will be 0-1 inputs for the respective arc-agi actions and 8-9 will contain (x, y) for ACTION6
        self.action_dim = (10,)
        # The raw game state will be represented as a 64x64 RGB image
        self.observation_dim = (3, 64, 64)

        self.prev_obs = None
        self.prev_info = None

    @property
    def as_arc(self) -> arc_agi.EnvironmentWrapper:
        assert isinstance(self._env, arc_agi.EnvironmentWrapper)
        return self._env
    
    @property
    def as_gym(self) -> gym.Env:
        assert isinstance(self._env, gym.Env)
        return self._env

    def unify_obs(
        self,
        obs: FrameDataRaw | np.ndarray
    ) -> np.ndarray:
        '''
        Maps the different state spaces from various benchmarks to a common representation

        :param obs: observation returned by env.step() or env.reset()
        :return observation (ndarray): rgb image of shape (3, 64, 64)
        '''
        if self.benchmark == self.Benchmark.ARC:
            #TODO: colormap indices
            # the map is constant for each game and [re]randomized each time a new game (not level) is started
            pass
        elif self.benchmark == self.Benchmark.ATARI:
            #TODO: downsize to 64x64
            # do we preserve aspect and randomize padding? or stretch to fit?
            pass

        raise NotImplementedError

    def reset(self) -> np.ndarray:
        if self.benchmark == self.Benchmark.ARC:
            obs = self.as_arc.reset()
            if (obs is None):
                raise RuntimeError("[Environment] Failed to reset ARC environment")
        elif self.benchmark == self.Benchmark.ATARI:
            obs, _ = self.as_gym.reset()
        return self.unify_obs(obs) # can consider adding an info dict if it's helpful later

    def step(self, action, xy=None) -> tuple[np.ndarray, bool, bool]:
        '''
        :param uint8 action: An index in the range [0, 7] that maps to the respective arc-agi action
        :param (uint16, uint16) xy: If action is ACTION6, xy should contain coordinates for the action
        :return observation (ndarray): RGB image of shape (3, 64, 64)
        :return done (bool): Whether episode has ended. If true, user needs to call reset()
        :return won (bool): Whether the episode was won (only true if done is true)
        '''
        if self.benchmark == self.Benchmark.ARC:
            if action == 6:
                assert xy is not None, "[Environment] xy coordinates must be provided for ACTION6"
                obs = self.as_arc.step(GameAction(action, ComplexAction), {"x": xy[0], "y": xy[1]})
            else:
                obs = self.as_arc.step(GameAction(action, SimpleAction))
            if (obs is None):
                raise RuntimeError("[Environment] Failed to step ARC environment")
            done = obs.state == GameState.WIN or obs.state == GameState.GAME_OVER
            won = obs.state == GameState.WIN
        elif self.benchmark == self.Benchmark.ATARI:
            obs, reward, terminated, truncated, _ = self.as_gym.step(action)
            done = terminated or truncated
            won = float(reward) > 0 and terminated #TODO: might need to verify this assumption

        return self.unify_obs(obs), done, won # can consider adding an info dict if it's helpful later


if __name__ == "__main__":
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    np.set_printoptions(threshold=sys.maxsize)

    arc = arc_agi.Arcade()

    # games = arc.get_environments()
    # for game in games:
    #     print(game)
    #     print(f"{game.game_id}: {game.title}")

    env = arc.make("ls20")
    frame = env.observation_space.frame[0]