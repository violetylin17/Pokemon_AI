import cv2
import numpy as np
import pyautogui
import time

# Define movement actions
actions = {
    0: "up",
    1: "down",
    2: "left",
    3: "right"
}

# Capture the game screen
def get_screen():
    x, y, width, height = 0, 100, 960, 860
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# Press key and animate movement
def press_key_with_animation(key, frame_count=3, delay=0.02):  # frame count how many caption during walking
    pyautogui.keyDown(key)
    for _ in range(frame_count):
        screen = get_screen()
        cv2.imshow("Pokémon Gold Screen", screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pyautogui.keyUp(key)
            exit()
        time.sleep(delay) ## key press duration
    pyautogui.keyUp(key)

# def press_key_with_animation(key, frame_count=2, delay=0.02):
#     for _ in range(frame_count):
#         pyautogui.keyDown(key)
#         screen = get_screen()
#         cv2.imshow("Pokémon Gold Screen", screen)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             pyautogui.keyUp(key)
#             exit()
#         time.sleep(delay)
#         pyautogui.keyUp(key)



# Main loop
if __name__ == "__main__":
    print("Starting screen capture with animation. Press 'q' to quit.")
    time.sleep(2) #2

    action_idx = 0
    while True:
        action = actions[action_idx % 4]
        print(f"Action: {action}")
        press_key_with_animation(action)
        action_idx += 1

        time.sleep(1)  # Now this actually controls the time between presses

        # Display one final frame for smoother transition
        screen = get_screen()
        cv2.imshow("Pokémon Gold Screen", screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
