import cv2
import numpy as np
import pyautogui
import time

goal_path = '/Users/violetlin/Documents/github/Pokemon_AI/env_images/cut_tiles/goal'
TILE_SIZE = 48
last_known_exit = None  # Agent memory

def get_screen():
    x, y, width, height = 60, 65, 480, 432  # Emulator window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def find_exit(screen, template_path=f'{goal_path}/downstairs.png'):
    global TILE_SIZE, last_known_exit

    template = cv2.imread(template_path)
    if template is None:
        print("❌ Exit template not found!")
        return None

    template = cv2.resize(template, (TILE_SIZE, TILE_SIZE))
    margin = TILE_SIZE
    cropped = screen[margin:-margin, margin:-margin]

    best_match = None
    best_val = -1
    for scale in [0.96, 0.98, 1.0, 1.02, 1.04]:
        resized_template = cv2.resize(template, (int(TILE_SIZE*scale), int(TILE_SIZE*scale)))
        if cropped.shape[0] < resized_template.shape[0] or cropped.shape[1] < resized_template.shape[1]:
            continue
        screen_blur = cv2.GaussianBlur(cropped, (3, 3), 0)
        res = cv2.matchTemplate(screen_blur, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_match = max_loc

    if best_val > 0.74:  # lowered threshold a bit
        match_x = best_match[0] + margin
        match_y = best_match[1] + margin
        col = round(match_x / TILE_SIZE)
        row = round(match_y / TILE_SIZE)
        last_known_exit = (row, col)
        return (row, col)

    return None

def get_grid_state(screen, template_path=f'{goal_path}/downstairs.png'):
    global last_known_exit
    height, width, _ = screen.shape
    grid_rows = height // TILE_SIZE
    grid_cols = width // TILE_SIZE
    grid_map = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    # Color-based detection of gray-ish areas (border)
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    # Lower brightness but still neutral color tones
    gray_mask = cv2.inRange(hsv, (0, 0, 50), (180, 60, 180))
    gray_mask = cv2.GaussianBlur(gray_mask, (3, 3), 0)

    for row in range(grid_rows):
        for col in range(grid_cols):
            tile = gray_mask[row*TILE_SIZE:(row+1)*TILE_SIZE, col*TILE_SIZE:(col+1)*TILE_SIZE]
            gray_density = np.sum(tile) / 255
            if gray_density > TILE_SIZE * TILE_SIZE * 0.2:
                grid_map[row, col] = 1  # Border / Inaccessible

    # Detect exit
    exit_coords = find_exit(screen, template_path)
    if not exit_coords and last_known_exit:
        print("🔁 Using last known exit.")
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
                color = (0, 0, 255)     # Red: inaccessible / border
                thickness = 2
            elif grid_map[row, col] == 2:
                color = (0, 255, 0)     # Green: exit
                thickness = 2
            else:
                color = (255, 255, 255) # White: walkable
                thickness = 1

            cv2.rectangle(overlay, top_left, bottom_right, color, thickness)
    return overlay

if __name__ == "__main__":
    print("🎮 Pokémon Grid Detect — Press 'q' to quit")
    time.sleep(2)

    while True:
        screen = get_screen()
        grid = get_grid_state(screen)
        grid_overlay = draw_grid_overlay(screen, grid)

        cv2.imshow("Pokémon Grid Overlay", grid_overlay)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
