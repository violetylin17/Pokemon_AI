import gymnasium as gym
from gymnasium import spaces
import cv2
import numpy as np
import pyautogui
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv

actions = {
    0: "up",    # Move up
    1: "down",  # Move down
    2: "left",  # Move left
    3: "right", # Move right
    4: "z",     # A button (VBA-M default)
    5: "x",     # B button (VBA-M default)
    6: "return",# Start
    7: "shift"  # Select
}

def press_key(key):
    pyautogui.keyDown(key)
    time.sleep(0.05)
    pyautogui.keyUp(key)

def get_screen():
    x, y, width, height = 60, 65, 480, 432  # Modify based on your setup 320, 288
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

player_templates = {
    "up": cv2.imread("char_image/player_up.png", cv2.IMREAD_GRAYSCALE),
    "down": cv2.imread("char_image/player_down.png", cv2.IMREAD_GRAYSCALE),
    "left": cv2.imread("char_image/player_left.png", cv2.IMREAD_GRAYSCALE),
    "right": cv2.imread("char_image/player_right.png", cv2.IMREAD_GRAYSCALE)
}

def get_grid_state(screen):
    grid = np.zeros((9, 10), dtype=np.uint8)
    for y in range(9):
        for x in range(10):
            tile = screen[y*16:(y+1)*16, x*16:(x+1)*16]
            if np.var(tile) > 40 or np.mean(tile) in range(50, 150):
                grid[y, x] = 1
            else:
                grid[y, x] = 0
    return grid

def get_player_position(screen, prev_position=(4,5)):
    best_match = None
    best_score = -1
    best_position = prev_position  # 若匹配失敗，返回上一次的位置

    for direction, template in player_templates.items():
        res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            best_match = direction
            best_position = (max_loc[0] // 16, max_loc[1] // 16)

    return best_position, best_match

class PokemonGoldGridEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=1, shape=(9, 10), dtype=np.uint8),
            "position": spaces.MultiDiscrete([10, 9])
        })
        self.state = None
        self.grid = None
        self.position = None
        self.reward = 0
        self.done = False
        self.steps = 0
        self.prev_position = None
        self.direction = None  # 用來記錄當前面向

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Ensures seeding works properly
        screen = get_screen()
        self.grid = get_grid_state(screen)
        self.position, self.direction = get_player_position(screen)
        self.prev_position = self.position
        self.state = {"grid": self.grid, "position": np.array(self.position)}
        self.reward = 0
        self.done = False
        self.steps = 0
        info = {"direction": self.direction, "steps": self.steps}
        return self.state, info  # Include useful info
        # return self.state, {}  # Gymnasium requires returning info as a dictionary


    def step(self, action):
        old_x, old_y = self.position
        target_direction = actions[action]  # 根據動作獲得目標方向

        # 先轉向
        if target_direction != self.direction:
            press_key(target_direction)  # 只轉向，不移動
            self.direction = target_direction  # 更新當前面向
            return self.state, -0.1, False, {}  # 轉向後不立即給正獎勵，避免原地轉圈

        # 方向正確，執行移動
        press_key(target_direction)
        screen = get_screen()
        self.grid = get_grid_state(screen)
        self.position, self.direction = get_player_position(screen)  # 更新位置與方向
        new_x, new_y = self.position

        # 計算獎勵
        self.reward = self.calculate_reward(action)

        # 判斷是否結束
        self.done = self.steps >= 1000
        self.steps += 1
        self.state = {"grid": self.grid, "position": np.array(self.position)}

            # Create info dictionary
        info = {
            "direction": self.direction,
            "steps": self.steps,
            "stuck": (new_x == old_x and new_y == old_y),
                }

        return self.state, self.reward, self.done, False, info  # False for `truncated`
        # return self.state, self.reward, self.done, False, {}  # `False` is for `truncated`

    def calculate_reward(self, action):
        new_x, new_y = self.position
        old_x, old_y = self.prev_position

        if new_x == old_x and new_y == old_y:
            return -1  # 移動失敗
        if self.grid[new_y, new_x] == 1:
            return 1  # 正常移動
        return -0.5  # 其他情況，稍微扣分
    
    def render(self):
        grid_display = self.grid * 255
        y, x = self.position
        grid_display[y, x] = 128
        cv2.imshow("Grid", cv2.resize(grid_display, (320, 288), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(1)

if __name__ == "__main__":
    pyautogui.click(100, 100)
    time.sleep(1)
    env = make_vec_env(PokemonGoldGridEnv, n_envs=1)
    # env = DummyVecEnv([lambda: PokemonGoldGridEnv()])
    model = PPO("MultiInputPolicy", env, verbose=1)
    # model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200) #50000
    model.save("pokemon_gold_grid_navigation_ppo")

    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
