import cv2
import numpy as np
import pyautogui
import time
import joblib

goal_path = '/Users/violetlin/Documents/github/Pokemon_AI/env_images/cut_tiles/goal'

TILE_SIZE = 48
last_known_exit = None  # Agent memory

def get_screen():
    x, y, width, height = 60, 65, 480, 432  # Emulator window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def find_exit(screen, template_path=f'{goal_path}/downstairs.png'):
    template = cv2.imread(template_path)
    if template is None:
        print("Exit template not found!")
        return None
    template = cv2.resize(template, (TILE_SIZE, TILE_SIZE))
    res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val > 0.8:  # Confidence threshold
        col = max_loc[0] // TILE_SIZE
        row = max_loc[1] // TILE_SIZE
        return (row, col)
    return None

def get_grid_state(screen, template_path=f'{goal_path}/downstairs.png'):
    global last_known_exit
    height, width, _ = screen.shape
    grid_rows = height // TILE_SIZE
    grid_cols = width // TILE_SIZE
    grid_map = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    edges = cv2.Canny(screen, 50, 150)

    for row in range(grid_rows):
        for col in range(grid_cols):
            tile = edges[row*TILE_SIZE:(row+1)*TILE_SIZE, col*TILE_SIZE:(col+1)*TILE_SIZE]
            edge_density = np.sum(tile) / 255
            if edge_density > 50:
                grid_map[row, col] = 1  # Inaccessible

    # Try to detect exit each frame
    exit_coords = find_exit(screen, template_path)
    if exit_coords:
        last_known_exit = exit_coords

    # Use last known if no new one is detected
    if last_known_exit:
        r, c = last_known_exit
        grid_map[r, c] = 2  # Mark exit

    return grid_map

def draw_grid_overlay(screen, grid_map):
    overlay = screen.copy()
    rows, cols = grid_map.shape
    for row in range(rows):
        for col in range(cols):
            top_left = (col * TILE_SIZE, row * TILE_SIZE)
            bottom_right = ((col + 1) * TILE_SIZE, (row + 1) * TILE_SIZE)

            if grid_map[row, col] == 1:
                color = (0, 0, 255)     # Red for inaccessible
                thickness = 2
            elif grid_map[row, col] == 2:
                color = (0, 255, 0)     # Green for exit
                thickness = 2
            else:
                color = (200, 200, 200) # Light gray for walkable
                thickness = 1

            cv2.rectangle(overlay, top_left, bottom_right, color, thickness)
    return overlay

if __name__ == "__main__":
    print("Starting screen capture. Press 'q' to quit.")
    time.sleep(5)

    while True:
        screen = get_screen()
        grid = get_grid_state(screen)
        grid_overlay = draw_grid_overlay(screen, grid)

        cv2.imshow("Pokémon Gold Grid", grid_overlay)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
