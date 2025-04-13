## day & night
import cv2
import numpy as np
import time
import pyautogui  # For screen capture

def get_screen():
    # Modify based on your setup 320, 288  ## mac: 60, 65, 480, 432 #window 3x 0,100,960,860
    x, y, width, height = 0, 100, 960, 860  
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def detect_time_of_day(screen):
    """Determine if it's day or night based on screen brightness."""
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    avg_brightness = np.mean(gray_screen)  # Compute average brightness
    return "night" if avg_brightness < 100 else "day"

def get_grid_state(screen):
    """Classify each tile into different categories based on grayscale intensity."""
    grid = np.zeros((9, 10), dtype=np.uint8)
    tile_size = 48
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    for y in range(9):
        for x in range(10):
            tile = gray_screen[y*tile_size:(y+1)*tile_size, x*tile_size:(x+1)*tile_size]
            gray_mean = np.mean(tile)
            
            if gray_mean > 200:  # Walkable ground (Should be White)
                grid[y, x] = 3  
            elif 170 < gray_mean <= 200:  # House Door (Should be Yellow)
                grid[y, x] = 1  
            elif 100 < gray_mean <= 170:  # NPCs, Signs (Should be Pink)
                grid[y, x] = 2  
            else:  # House Walls, Trees (Should be Black)
                grid[y, x] = 0  
                
    return grid

if __name__ == "__main__":
    print("Starting screen capture with dynamic grid colors. Press 'q' to quit.")
    time.sleep(5)  # Time to focus emulator
    
    while True:
        screen = get_screen()  # Capture the screen
        grid = get_grid_state(screen)  # Process screen into a grid
        time_of_day = detect_time_of_day(screen)  # Detect if it's day or night
        
        # Display original screen
        cv2.imshow("Pokémon Gold Screen", screen)

        # Define colors based on time of day
        if time_of_day == "day":
            colors = {
                0: (0, 0, 0),       # Black for Trees, House Walls
                1: (0, 255, 255),   # Yellow for House Doors
                2: (255, 0, 255),   # Pink for NPCs, Signs
                3: (255, 255, 255)  # White for Walkable Paths
            }
        else:  # Night Mode with Blue Gradient
            colors = {
                0: (0, 0, 0),       # Black for Trees, House Walls
                1: (0, 255, 255),   # Yellow for House Doors
                2: (255, 0, 255),   # Pink for NPCs, Signs
                3: (255, 255, 255)  # White for Walkable Paths
            }

        # Create an RGB visualization of the grid
        grid_display = np.zeros((9, 10, 3), dtype=np.uint8)
        for y in range(9):
            for x in range(10):
                grid_display[y, x] = colors[grid[y, x]]

        # Scale up for visibility
        grid_scaled = cv2.resize(grid_display, (320, 288), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Grid", grid_scaled)

        # Proper exit condition
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
