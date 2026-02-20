import colorsys
import numpy as np

def hex_to_rgb(hex_str: str):
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return r, g, b

def rgb_to_hex(r: int, g: int, b: int) -> str:
    if not all(0 <= v <= 255 for v in (r, g, b)):
        raise ValueError("RGB components must be in 0–255")
    return f"#{r:02X}{g:02X}{b:02X}FF"

def hex_to_hsv(hex_str: str):
    r, g, b = hex_to_rgb(hex_str)
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return h, s, v

def hsv_to_hex(h: float, s: float, v: float) -> str:
    """
    Convert HSV to #RRGGBB hex.
    H in degrees [0, 360), S and V in [0, 1].
    """
    if not (0.0 <= h <= 1.0 and 0.0 <= s <= 1.0 and 0.0 <= v <= 1.0):
        raise ValueError("H, S and V must be in [0, 1]")
    r_f, g_f, b_f = colorsys.hsv_to_rgb(h, s, v)
    # Scale back to 0–255 and round
    r = int(round(r_f * 255))
    g = int(round(g_f * 255))
    b = int(round(b_f * 255))
    return rgb_to_hex(r, g, b)

HUE = np.array([0.420, 0.898, 0.009, 0.580, 0.144, 0.077, 0.960, 0.300, 0.767]) # reuse 0 for 0-5, 1 and 3 need to be repeated once
SAT = np.array([0.747, 0.518, 0.803, 0.882, 0.436, 1.000, 0.894, 0.877, 0.765, 0.598])
VALUE = np.array([1.000, 0.800, 0.600, 0.400, 0.200, 0.000, 0.898, 1.000, 0.976, 1.000, 0.945, 1.000, 1.000, 0.573, 0.800, 0.839])
I_TO_J = [0, 6, 8, 9, 11, 12, 13, 14, 15]

class ColorMapGenerator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        
    def generate(self) -> dict[int, str]:
        # randomize hues while keeping certain relationships according to arc agi's default color map
        hues = np.mod(self.rng.random() + HUE, 1.0)
        random_sat = self.rng.random()
        color_map = np.empty((16, 3))
        order = np.concatenate((np.array([0]), self.rng.permutation(np.arange(1, 9))))
        for k in range(9):
            i = order[k]
            j = I_TO_J[k]
            if k == 0:
                for dj in range(6):
                    color_map[j+dj][0] = hues[i]
                    color_map[j+dj][1] = random_sat
            else:
                color_map[j][0] = hues[i]
                color_map[j][1] = SAT[j-6]
                if k == 1:
                    color_map[j+1][0] = hues[i]
                    color_map[j+1][1] = SAT[j+1-6]
                elif k == 3:
                    color_map[j+1][0] = np.mod(hues[i]-0.04, 1.0)
                    color_map[j+1][1] = SAT[j+1-6]
        color_map[:, 2] = VALUE
        #TODO: can choose to add small jitter to hsv values, but check the baseline for now

        hex_color_map = {i: hsv_to_hex(*color_map[i]) for i in range(16)}
        return hex_color_map

def visualize_color_map(color_map: dict[int, str]):
    import matplotlib.pyplot as plt
    colors = [hex_to_rgb(color_map[i]) for i in range(16)]
    plt.figure(figsize=(8, 2))
    plt.bar(range(16), [1]*16, color=[(r/255, g/255, b/255) for r, g, b in colors])
    plt.xticks(range(16))
    plt.yticks([])
    plt.title("Generated Color Map")
    plt.show()

if __name__ == "__main__":
    COLOR_MAP: dict[int, str] = {
        0: "#FFFFFFFF",  # White
        1: "#CCCCCCFF",  # Off-white
        2: "#999999FF",  # neutral Light
        3: "#666666FF",  # neutral
        4: "#333333FF",  # Off Black
        5: "#000000FF",  # Black
        6: "#E53AA3FF",  # Magenta
        7: "#FF7BCCFF",  # Magenta Light
        8: "#F93C31FF",  # Red
        9: "#1E93FFFF",  # Blue
        10: "#88D8F1FF",  # Blue Light
        11: "#FFDC00FF",  # Yellow
        12: "#FF851BFF",  # Orange
        13: "#921231FF",  # Maroon
        14: "#4FCC30FF",  # Green
        15: "#A356D6FF",  # Purple
    }
    generator = ColorMapGenerator()
    color_map = generator.generate()
    visualize_color_map(color_map)