# import cv2
# import numpy as np
# import pyautogui
# import time

# #Press 'q' to exit the capture window.
# def get_screen():
#     # Adjust these values to match your emulator window position and size
#     x, y, width, height = 60, 65, 320, 288  # Modify based on your setup
#     screenshot = pyautogui.screenshot(region=(x, y, width, height))
#     frame = np.array(screenshot)  # No grayscale conversion
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV for color
#     # frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2GRAY)  # Convert to grayscale
#     return frame

# def main():
#     print("Starting screen capture test...")
#     time.sleep(2)  # Give time to switch to emulator window

#     while True:
#         frame = get_screen()
#         cv2.imshow("Screen Capture Test", frame)  # Display captured screen

#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
#             break

#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import pyautogui
import time

def get_screen():
    x, y, width, height = 0, 100, 960, 860  # Modify based on your setup 320, 288  ## mac: 60, 65, 480, 432 #window 3x 0,100,960,860
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    print("Starting screen capture. Press Ctrl+C to stop.")
    time.sleep(2)  # Give time to focus VBA-M/OpenEmu
    while True:
        screen = get_screen()
        cv2.imshow("Pokémon Gold Screen", screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cv2.destroyAllWindows()