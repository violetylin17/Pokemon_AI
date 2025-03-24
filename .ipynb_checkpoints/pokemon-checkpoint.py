import gym
from gym import spaces
import cv2
import numpy as np
import pyautogui
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv




# Define Game Boy actions (adjust based on emulator key bindings)
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
    # Adjust region to match your emulator window (e.g., 160x144 for Game Boy)
    screenshot = pyautogui.screenshot(region=(65, 35, 320, 288))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)

class PokemonRedEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(144, 160, 1), dtype=np.uint8)
        self.state = None
        self.reward = 0
        self.done = False
        self.steps = 0

    def reset(self):
        # Manually reset the emulator or use a save state
        self.state = get_screen()
        self.reward = 0
        self.done = False
        self.steps = 0
        return self.state

    def step(self, action):
        press_key(actions[action])
        self.state = get_screen()
        self.reward = self.calculate_reward()
        self.done = self.check_done()
        self.steps += 1
        return self.state, self.reward, self.done, {}

    def calculate_reward(self):
        return 1  # Basic reward (improve later)

    def check_done(self):
        return self.steps >= 1000  # End after 1000 steps

    def render(self):
        cv2.imshow("Game", self.state)
        cv2.waitKey(1)

# # Test and train
# if __name__ == "__main__":
#     # Focus the emulator window
#     pyautogui.click(100, 100)  # Click inside emulator
#     time.sleep(1)

#     # Test environment
#     env = PokemonRedEnv()
#     obs = env.reset()
#     for _ in range(100):
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         env.render()
#         if done:
#             obs = env.reset()
#     cv2.destroyAllWindows()

#     # Train AI
#     env = DummyVecEnv([lambda: PokemonRedEnv()])
#     model = PPO("CnnPolicy", env, verbose=1)
#     model.learn(total_timesteps=10000)
#     model.save("pokemon_red_ppo")
