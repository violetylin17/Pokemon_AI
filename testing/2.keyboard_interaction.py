import cv2
import numpy as np
import pyautogui
import time
# import mss

## add in press_key & actions
actions = {
    0: "up",
    1: "down",
    2: "left",
    3: "right"
}

def press_key(key):
    pyautogui.keyDown(key)
    time.sleep(0.02)  # key press duration
    pyautogui.keyUp(key)
    
def get_screen():
    x, y, width, height = 0, 100, 960, 860  # Modify based on your setup 320, 288  ## mac: 60, 65, 480, 432 #window 3x 0,100,960,860
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# # tackling delay capture WITH IMPORT MSS
# def get_screen(): 
#     x, y, width, height = 0, 100, 960, 860
#     with mss.mss() as sct:
#         monitor = {"top": y, "left": x, "width": width, "height": height}
#         screenshot = sct.grab(monitor)
#         img = np.array(screenshot)
#         return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

if __name__ == "__main__":
    print("Starting screen capture with grid and movement (480x432). Press 'q' to quit.")
    time.sleep(5)  # Time to focus emulator
    
    action_idx = 0  # Simple cycling AI for testing
    while True:

        # AI movement: Cycle through actions
        action = actions[action_idx % 4]
        print(f"Action: {action}")
        press_key(action)
        action_idx += 1

        # Wait before next action
        time.sleep(1)  # Add this to control loop speed (adjust as needed)

        # Display original screen (BGR)
        screen = get_screen()
        cv2.imshow("Pokémon Gold Screen", screen)

       
        if cv2.waitKey(100) & 0xFF == ord('q'):  #weitkey for screen capture maybe?
            break
    
    cv2.destroyAllWindows()
