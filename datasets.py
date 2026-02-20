import arc_agi
from arcengine import GameAction, FrameDataRaw, GameState, SimpleAction, ComplexAction
import gymnasium as gym

import numpy as np
import torch
import torch.nn.functional as F

import math
from enum import Enum

from color import ColorMapGenerator, hex_to_rgb

class Environment:
    class Benchmark(Enum):
        ARC = "arc"
        ATARI = "atari"

    def __init__(
        self,
        benchmark: str,
        env: arc_agi.EnvironmentWrapper | gym.Env,
        seed=None,
    ):
        if (benchmark not in self.Benchmark):
            raise ValueError(f"[Environment] Unsupported benchmark: {benchmark}")
        
        self._env = env
        self.benchmark = self.Benchmark(benchmark)
        self.rng = np.random.default_rng(seed)

        # Indices 0-7 will be 0-1 inputs for the respective arc-agi actions and 8-9 will contain (x, y) for ACTION6
        self.action_dim = (10,)
        # The raw game state will be represented as a 64x64 RGB image
        self.observation_dim = (3, 64, 64)

        self.obs = None
        self.info = None
        self.prev_obs = None
        self.prev_info = None

        if self.benchmark == self.Benchmark.ARC:
            self.map_gen = ColorMapGenerator()
            self.color_map = self.generate_color_map()
            self.target_size = self.rng.integers(64, 161)

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
        :return observation (ndarray): rgb image of shape (3, 210, 160)
        '''
        if self.benchmark == self.Benchmark.ARC:
            # the map and size is constant for each game and [re]randomized each time a new game (not level) is started
            assert isinstance(obs, FrameDataRaw)

            observation = obs.frame[-1]

            # colormap indices
            indices = np.searchsorted(range(16), observation)
            values = np.array(list(self.color_map.values()))
            observation = values[indices]

            observation = observation.transpose(2, 0, 1) # to CHW

            # randomly scale up image to at most 160x160
            observation = F.interpolate(
                torch.from_numpy(observation).to(torch.float32).unsqueeze(0), 
                size = self.target_size,
                mode="nearest-exact"
            ).squeeze(0).numpy()

            # pad to 210x160
            t_pad = self.rng.integers(0, 210-self.target_size + 1)
            b_pad = 210-self.target_size - t_pad
            l_pad = self.rng.integers(0, 160-self.target_size + 1)
            r_pad = 160-self.target_size - l_pad
            observation = np.pad(observation, ((0, 0), (t_pad, b_pad), (l_pad, r_pad)), mode='constant', constant_values=self.rng.integers(0, 256))

            return observation
        
        if self.benchmark == self.Benchmark.ATARI:
            assert isinstance(obs, np.ndarray)

            H, _, _ = obs.shape
            if H > 210:
                obs = obs[H//2 - 210//2 : H//2 + 210//2, :, :]
            observation = obs.transpose(2, 0, 1) # to CHW
            return observation
        
    def generate_color_map(self):
        assert self.benchmark == self.Benchmark.ARC
        color_map = self.map_gen.generate()
        return {i: np.array(hex_to_rgb(color_map[i])) for i in range(16)}

    def reset(self) -> np.ndarray:
        if self.benchmark == self.Benchmark.ARC:
            obs = self.as_arc.reset()
            if (obs is None):
                raise RuntimeError("[Environment] Failed to reset ARC environment")
            self.color_map = self.generate_color_map()
            self.target_size = self.rng.integers(64, 161)
        elif self.benchmark == self.Benchmark.ATARI:
            obs, _ = self.as_gym.reset()
        return self.unify_obs(obs) # can consider adding an info dict if it's helpful later

    def step(self, action, xy=None) -> tuple[np.ndarray, bool, bool]:
        '''
        :param uint8 action: An index in the range [0, 7] that maps to the respective arc-agi action
        :param (uint16, uint16) xy: If action is ACTION6, xy should contain coordinates for the action
        :return observation (ndarray): RGB image of shape (3, 210, 160)
        :return done (bool): Whether episode has ended. If true, user needs to call reset()
        :return won (bool): Whether the level was won (can be true even if done is false)
        '''
        if self.benchmark == self.Benchmark.ARC:
            if action == 6:
                assert xy is not None, "[Environment] xy coordinates must be provided for ACTION6"
                obs = self.as_arc.step(GameAction(action, ComplexAction), {"x": xy[0], "y": xy[1]})
            else:
                obs = self.as_arc.step(GameAction(action, SimpleAction))
            if (obs is None):
                raise RuntimeError("[Environment] Failed to step ARC environment")
            if obs.full_reset:
                self.color_map = self.generate_color_map()
                self.target_size = self.rng.integers(64, 161)
            info = {
                "levels_completed": obs.levels_completed,
            }
            done = obs.state == GameState.WIN or obs.state == GameState.GAME_OVER
            won = (obs.state == GameState.WIN or 
                   (obs.levels_completed > self.prev_info["levels_completed"] if self.prev_info else False))
        elif self.benchmark == self.Benchmark.ATARI:
            obs, reward, terminated, truncated, info = self.as_gym.step(action)
            done = terminated or truncated
            won = float(reward) > 0 and terminated #TODO: might need to verify this assumption

        self.prev_obs = self.obs
        self.prev_info = self.info
        self.obs = self.unify_obs(obs)
        self.info = info
        return self.obs, done, won

if __name__ == "__main__":
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    np.set_printoptions(threshold=sys.maxsize)

    import ale_py
    gym.register_envs(ale_py)

    arc = arc_agi.Arcade()

    # games = arc.get_environments()
    # for game in games:
    #     print(game)
    #     print(f"{game.game_id}: {game.title}")


    def render_3chw_image(arr, block=True):
        """
        Render a 3xHxW numpy array as an RGB image.
        
        Args:
            arr: numpy array of shape (3, H, W) with values in [0, 1] or [0, 255]
        
        Returns:
            matplotlib image plot
        """
        # Normalize to [0, 1] if needed (handles uint8 or float)
        if arr.max() > 1.0:
            arr = arr.astype(np.float32) / 255.0
        
        # Transpose to HWC for matplotlib
        img = np.transpose(arr, (1, 2, 0))
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show(block=block)
        
    env = arc.make("ls20")
    assert env is not None
    test_arc = Environment("arc", env)
    obs = env.reset()
    assert obs is not None
    render_3chw_image(test_arc.unify_obs(obs))

    # env = gym.make("ALE/Casino-v5")
    # test_atari = Environment("atari", env)
    # obs, _ = env.reset()
    # render_3chw_image(test_atari.unify_obs(obs))
