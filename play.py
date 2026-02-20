import sys
import argparse
import arc_agi
from arc_agi.rendering import render_frames
from arcengine import GameAction, GameState

from color import ColorMapGenerator

class InputHandler:
    def __init__(self):
        if sys.platform == "win32":
            import msvcrt
            self.msvcrt = msvcrt

        self.mapping = {
            'r': GameAction.RESET,
            'w': GameAction.ACTION1,
            's': GameAction.ACTION2,
            'a': GameAction.ACTION3,
            'd': GameAction.ACTION4,
            ' ': GameAction.ACTION5,
            # TODO: skip ACTION6 for now, but try to implement later
            'u': GameAction.ACTION7,
        }
    
    def getch(self):
        if sys.platform == "win32":
            if self.msvcrt.kbhit():
                return self.msvcrt.getch().decode('utf-8', errors='ignore')
            return ''
        else:
            import termios
            import tty

            fd = sys.stdin.fileno()
            try:
                old_settings = tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
    
    def input_to_action(self, ch):
        return self.mapping.get(ch, None)


def play_arc_game(game_id, input_handler, random_color=False):
    arc = arc_agi.Arcade()

    if random_color:
        map_gen = ColorMapGenerator()
        color_map = map_gen.generate()
        custom_renderer = lambda steps, frame_data: render_frames(
            steps=steps,
            frame_data=frame_data,
            scale=4,
            color_map=color_map,
        )

    env = arc.make(game_id, render_mode="human", renderer=custom_renderer if random_color else None)
    if env is None:
        raise ValueError(f'Failed to create environment for "{game_id}". Does this game exist?')
    while True:
        ch = input_handler.getch()
        if ch == 'q':
            break
        action = input_handler.input_to_action(ch)
        if action is not None:
            obs = env.step(action)
            if obs is None:
                raise RuntimeError("[play_arc_game] Failed to step environment")
            if random_color and obs.full_reset: # first frame after reset renders wrongly, but good enough for debugging. won't affect model training
                color_map = map_gen.generate()
            #TODO: format print output if want nicer debugging
            print(obs)
            print(arc.get_scorecard())
            if obs.state == GameState.WIN:
                print("Game won!")
                break
            elif obs.state == GameState.GAME_OVER:
                print("Game over!")
                break


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("-b", "--benchmark", type=str, default="arc", help="Benchmark to run (arc or atari)")
    p.add_argument("-g", "--game_id", type=str, default="ls20", help="Name of game to run (e.g. ls20 for ARC or Adventure for Atari)")
    p.add_argument("--random-color", action="store_true", help="Test random color mapping for rendering (only applicable for ARC benchmark)")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    input_handler = InputHandler()

    if args.benchmark.lower() == "arc":
        play_arc_game(args.game_id, input_handler, random_color=args.random_color)
    elif args.benchmark.lower() == "atari":
        raise NotImplementedError("Atari benchmark not implemented yet")
