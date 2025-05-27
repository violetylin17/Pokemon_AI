import cv2
import numpy as np
import pyautogui
import time
import os

goal_path = '/Users/violetlin/Documents/github/Pokemon_AI/env_images/cut_tiles/goal'
TILE_SIZE = 48
last_known_exit = None  # Agent memory

def get_screen():
    x, y, width, height = 60, 65, 480, 432  # Emulator window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def preprocess_for_matching(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def find_exit(screen, template_path=f'{goal_path}/downstairs.png'):
    global TILE_SIZE

    template = cv2.imread(template_path)
    if template is None:
        print("Exit template not found!")
        return None

    template = cv2.resize(template, (TILE_SIZE, TILE_SIZE))

    # Optional: restrict to center area
    margin = TILE_SIZE
    cropped_screen = screen[margin:-margin, margin:-margin]

    best_match = None
    best_val = -1
    for scale in [0.98, 1.0, 1.02]:
        scaled_template = cv2.resize(template, (int(TILE_SIZE*scale), int(TILE_SIZE*scale)))
        if cropped_screen.shape[0] < scaled_template.shape[0] or cropped_screen.shape[1] < scaled_template.shape[1]:
            continue

        blurred_screen = cv2.GaussianBlur(cropped_screen, (3, 3), 0)
        res = cv2.matchTemplate(blurred_screen, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_val:
            best_val = max_val
            best_match = max_loc

    if best_val > 0.78:
        # Offset match location to original screen
        match_x = best_match[0] + margin
        match_y = best_match[1] + margin

        # Snap to closest TILE center to eliminate pixel drift
        col = round(match_x / TILE_SIZE)
        row = round(match_y / TILE_SIZE)

        return (row, col)

    return None


def get_grid_state(screen, template_path=f'{goal_path}/downstairs.png'):
    height, width, _ = screen.shape
    grid_rows = height // TILE_SIZE
    grid_cols = width // TILE_SIZE
    grid_map = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    # Use color-based detection for dark borders (outside gameplay area)
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))  # dark gray/black
    dark_mask = cv2.GaussianBlur(dark_mask, (3, 3), 0)

    for row in range(grid_rows):
        for col in range(grid_cols):
            tile = dark_mask[row*TILE_SIZE:(row+1)*TILE_SIZE, col*TILE_SIZE:(col+1)*TILE_SIZE]
            density = np.sum(tile) / 255
            if density > TILE_SIZE * TILE_SIZE * 0.2:  # 20% dark pixels
                grid_map[row, col] = 1  # Border or inaccessible

    # Detect exit
    exit_coords = find_exit(screen, template_path)
    if not exit_coords and last_known_exit:
        print("🔁 Using last known exit position.")
        exit_coords = last_known_exit

    if exit_coords:
        r, c = exit_coords
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
                color = (0, 0, 255)     # Red for inaccessible or border
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
    print("🎮 Starting Pokémon Gold grid detection. Press 'q' to quit.")
    time.sleep(3)

    while True:
        screen = get_screen()
        grid = get_grid_state(screen)
        grid_overlay = draw_grid_overlay(screen, grid)

        cv2.imshow("Pokémon Gold Grid", grid_overlay)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
