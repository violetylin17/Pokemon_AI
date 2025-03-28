import cv2
import numpy as np
import pyautogui
import time

## add in press_key & actions
# actions = {
#     0: "up",
#     1: "down",
#     2: "left",
#     3: "right"
# }

# def press_key(key):
#     pyautogui.keyDown(key)
#     time.sleep(0.05)
#     pyautogui.keyUp(key)

def get_screen():
    x, y, width, height = 60, 65, 480, 432  # Modify based on your setup
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def get_grid_state(screen):
    grid = np.zeros((9, 10), dtype=np.uint8)
    tile_size = 48
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)  # Convert to grayscale

    for y in range(9):
        for x in range(10):
            tile = gray_screen[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
            gray_mean = np.mean(tile)  # Compute mean grayscale intensity
            
            # Define rules based on grayscale values
            if 180 < gray_mean < 230:  # Grass and walkable areas
                grid[y, x] = 1  # Accessible
            elif 100 < gray_mean < 140:  # House doors, NPCs
                grid[y, x] = 2  # Special objects (doors/NPCs)
            else:
                grid[y, x] = 0  # Inaccessible (trees, walls, obstacles)
                
    return grid
# def get_grid_state(screen):
#     grid = np.zeros((9, 10), dtype=np.uint8)
#     tile_size = 48
#     for y in range(9):
#         for x in range(10):
#             tile = screen[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
#             b_mean = np.mean(tile[:, :, 0])
#             g_mean = np.mean(tile[:, :, 1])
#             r_mean = np.mean(tile[:, :, 2])
            
#             # Define rules based on extracted color statistics
#             if (200 < r_mean < 235 and 220 < g_mean < 250 and 200 < b_mean < 225):
#                 grid[y, x] = 1  # Accessible (path/grass)
#             else:
#                 grid[y, x] = 0  # Inaccessible (trees or obstacles)
                
#     return grid

if __name__ == "__main__":
    print("Starting screen capture with grid (480x432, BGR). Press 'q' to quit.")
    time.sleep(5)  # Time to focus emulator
    
    while True:
        screen = get_screen()
        grid = get_grid_state(screen)
        
        # Display original screen (BGR, not grayscale)
        cv2.imshow("Pokémon Gold Screen", screen)

        # Define colors for different terrain types
        grid_display = np.zeros((9, 10, 3), dtype=np.uint8)  # Create an RGB image for better visualization

        for y in range(9):
            for x in range(10):
                if grid[y, x] == 1:  # Walkable areas (grass, path)
                    grid_display[y, x] = (255, 255, 255)  # White
                elif grid[y, x] == 2:  # Special objects (doors, NPCs)
                    grid_display[y, x] = (0, 255, 255)  # Yellow
                else:  # Inaccessible (trees, walls, obstacles)
                    grid_display[y, x] = (0, 0, 0)  # Black
        
        # Scale up for visibility
        grid_scaled = cv2.resize(grid_display, (320, 288), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Grid", grid_scaled)

        # Proper exit condition
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()  # Ensure all windows close properly

    
# if __name__ == "__main__":
#     print("Starting screen capture with grid. Align the emulator window. Focus on the emulator and press 'a' to start AI movement, or press 'q' in this console to quit.")
#     time.sleep(1)

#     waiting_for_start_signal = True
#     while waiting_for_start_signal:
#         screen = get_screen()
#         grid = get_grid_state(screen)

#         # Display original screen (BGR)
#         cv2.imshow("Pokémon Gold Screen", screen)

#         # Display grid
#         grid_display = grid * 255
#         grid_scaled = cv2.resize(grid_display, (480, 432), interpolation=cv2.INTER_NEAREST)
#         cv2.imshow("Grid", grid_scaled)

#         key = cv2.waitKey(100)
#         if key == ord('q'):
#             waiting_for_start_signal = False
#             break
#         # We no longer check for 'a' here for the start signal

#         # Instead, we'll rely on a separate mechanism (e.g., a console input)
#         user_input = input("Press 'a' in the emulator window to start AI, or type 'quit' here: ").lower()
#         if user_input == 'a':
#             waiting_for_start_signal = False
#             print("Starting AI movement...")
#         elif user_input == 'quit':
#             waiting_for_start_signal = False
#             break

#     if not waiting_for_start_signal:
#         action_idx = 0  # Simple cycling AI for testing
#         while True:

            
#             screen = get_screen()
#             grid = get_grid_state(screen)

#             # Display original screen (BGR)
#             cv2.imshow("Pokémon Gold Screen", screen)

#             # Display grid
#             grid_display = grid * 255
#             grid_scaled = cv2.resize(grid_display, (480, 432), interpolation=cv2.INTER_NEAREST)
#             cv2.imshow("Grid", grid_scaled)

#             # AI movement: Cycle through actions
#             # Inside the second while loop (AI movement):
#             action = actions[action_idx % 4]
#             print(f"Action: {action}")
#             emulator_title = "Your Emulator Window Title Here"  # Replace with the actual title
#             emulator_windows = pyautogui.getWindowsWithTitle(emulator_title)
#             if emulator_windows:
#                 emulator_window = emulator_windows[0]  # Assuming only one window with that title
#                 emulator_window.activate()
#                 time.sleep(0.1) # Give it a moment to activate
#                 press_key(action)
#             else:
#                 print(f"Emulator window with title '{emulator_title}' not found.")
#             action_idx += 1
#             # action = actions[action_idx % 4]
#             # print(f"Action: {action}")
#             # press_key(action)
#             # action_idx += 1

#             if cv2.waitKey(100) & 0xFF == ord('q'):
#                 break

#     cv2.destroyAllWindows()