import cv2
import numpy as np
import pyautogui
import time
import joblib

# Load classifier
# clf = joblib.load("/Users/violetlin/Documents/github/Pokemon_AI/tile_classifier.pkl")

def get_screen():
    x, y, width, height = 60, 65, 480, 432  # Emulator window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def get_grid_state(screen):
    grid = np.zeros((9, 10), dtype=np.uint8)
    tile_size = 48
    for y in range(9):
        for x in range(10):
            tile = screen[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
            b_mean = np.mean(tile[:, :, 0])
            g_mean = np.mean(tile[:, :, 1])
            r_mean = np.mean(tile[:, :, 2])
            # Your custom rules
            if (y > 2 and y < 7 and x > 1 and x < 8) or g_mean > 100:  # Example: middle rows or green
                grid[y, x] = 1  # Accessible
            else:
                grid[y, x] = 0  # Inaccessible
    return grid
    
# def get_grid_state(screen):
#     grid = np.zeros((9, 10), dtype=np.uint8)
#     tile_size = 48
#     target_size = (32, 32)  # Classifier expects 32x32 tiles

#     for y in range(9):
#         for x in range(10):
#             # Extract 48x48 tile
#             tile = screen[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
#             if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
#                 grid[y, x] = 3  # Default to inaccessible if tile is invalid
#                 continue
#             # Resize to 32x32 for classifier
#             tile_resized = cv2.resize(tile, target_size, interpolation=cv2.INTER_AREA)
#             # Classify tile
#             label = clf.predict([tile_resized.flatten()])[0]
#             grid[y, x] = label
            
#     return grid

if __name__ == "__main__":
    print("Starting screen capture (480x432, 9x10 grid). Press 'q' to quit.")
    time.sleep(5)  # Time to focus emulator
    
    while True:
        screen = get_screen()
        grid = get_grid_state(screen)
        
        # Display original screen
        cv2.imshow("Pokémon Gold Screen", screen)

        # Create colored grid visualization
         # Display grid
        grid_display = grid * 255  # 0=black, 1=white
        grid_scaled = cv2.resize(grid_display, (320, 288), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Grid", grid_scaled)
        # grid_display = np.zeros((9, 10, 3), dtype=np.uint8)
        # for y in range(9):
        #     for x in range(10):
        #         if grid[y, x] == 0:  # Character
        #             grid_display[y, x] = (255, 0, 0)  # Red
        #         elif grid[y, x] == 1:  # Accessible
        #             grid_display[y, x] = (255, 255, 255)  # White
        #         elif grid[y, x] == 2:  # Goal
        #             grid_display[y, x] = (0, 255, 255)  # Yellow
        #         else:  # Inaccessible
        #             grid_display[y, x] = (0, 0, 0)  # Black
        
        # # Scale up grid for visibility
        # grid_scaled = cv2.resize(grid_display, (320, 288), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow("Grid", grid_scaled)

        # Exit on 'q'
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()