import arc_agi
from arcengine import GameAction
import numpy

class Environment:
    '''
    An environment wrapper to standardize the interface for different datasets e.g. gymnasium, arc-agi
    '''
    def __init__(self, env: arc_agi.EnvironmentWrapper):
        self.env = env

    @property
    def action_dim(self):
        return len(self.env.action_space)

    @property
    def observation_dim(self):
        return len(self.env.observation_space.frame), *self.env.observation_space.frame[0].shape

    def reset(self):
        obs = self.env.reset()
        return obs # probably need to process to be consistent with gymnasium

    def step(self, action):
        pass

class ARCAGIEnv(Environment):
    def __init__(self, env):
        super().__init__(env)
    

if __name__ == "__main__":
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    np.set_printoptions(threshold=sys.maxsize)

    arc = arc_agi.Arcade()

    # games = arc.get_environments()
    # for game in games:
    #     print(f"{game.game_id}: {game.title}")

    env = arc.make("ls20", render_mode="terminal")
    frame = env.observation_space.frame[0]
    norm_arr = (frame - frame.min()) / (frame.max() - frame.min())

    plt.figure(figsize=(8, 8))
    plt.imshow(norm_arr, cmap='tab20', interpolation='nearest')  # 'tab20' gives 20 distinct colors
    plt.colorbar()  # Shows value-to-color mapping
    plt.axis('off')
    plt.show()