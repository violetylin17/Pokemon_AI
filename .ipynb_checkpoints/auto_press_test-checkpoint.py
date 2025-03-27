from applescript import run

script = """
tell application "OpenEmu"
    activate
end tell
"""

try:
    result = run(script)
    print(f"AppleScript result: {result}")
    print("OpenEmu should have been activated.")
except Exception as e:
    print(f"Error running AppleScript: {e}")


# import pyautogui
# import time

# def press_key_long(key):
#     print(f"Pressing and holding key: {key}")
#     pyautogui.keyDown(key)
#     time.sleep(0.5)  # Hold the key for 0.5 seconds
#     pyautogui.keyUp(key)
#     time.sleep(0.2)  # Short delay after release

# if __name__ == "__main__":
#     print("Focus on your OpenEmu window. Key presses will start in 5 seconds...")
#     time.sleep(5)

#     test_keys = ['a', 'enter', 'w', 'a', 's', 'd', 'up', 'down', 'right', 'left']

#     for key in test_keys:
#         press_key_long(key)

#     print("Testing complete.")