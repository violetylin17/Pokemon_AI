import cv2
import numpy as np
import pyautogui
import time
import math

goal_path = '/Users/violetlin/Documents/github/Pokemon_AI/env_images/cut_tiles/goal'
TILE_SIZE = 48
last_known_exit = None  # Agent memory

def get_screen():
    x, y, width, height = 60, 65, 480, 432  # Emulator window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
# --- Border Detection Functions ---
def detect_game_border(screen):
    # 1. Initial crop (optional)
    cropped = screen[5:screen.shape[0]-5, 5:screen.shape[1]-5]

    # 2. Edge Detection
    edges = cv2.Canny(cropped, 50, 150)

    # 3. Morphological operations (optional)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    eroded_edges = cv2.erode(dilated_edges, kernel, iterations=2)

    # 4. Find largest contour
    contours, _ = cv2.findContours(eroded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Adjust coordinates back to the original screen
        return x + 5, y + 5, w, h
    return None

def crop_to_game_area(screen, border):
    if border:
        x, y, w, h = border
        return screen[y:y+h, x:x+w].copy()
    return screen

# --- Exit Detection ---
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

# --- Grid State Detection ---
def get_grid_state(screen, game_border=None, template_path=f'{goal_path}/downstairs.png'):
    global last_known_exit
    if game_border:
        x, y, w, h = game_border
        game_screen = screen[y:y+h, x:x+w].copy()
    else:
        game_screen = screen.copy()

    height, width, _ = game_screen.shape
    grid_rows = math.ceil(height / TILE_SIZE) # 使用 math.ceil() 向上取整
    grid_cols = math.ceil(width / TILE_SIZE)  # 使用 math.ceil() 向上取整
    grid_map = np.zeros((int(grid_rows), int(grid_cols)), dtype=np.uint8) # 確保 grid_map 的 shape 是整數

    for row in range(int(grid_rows)):
        for col in range(int(grid_cols)):
            tile_y_start = row * TILE_SIZE
            tile_y_end = min((row + 1) * TILE_SIZE, height) # 確保不超出邊界
            tile_x_start = col * TILE_SIZE
            tile_x_end = min((col + 1) * TILE_SIZE, width)  # 確保不超出邊界

            tile = game_screen[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
            if tile.size > 0: # 確保 tile 不是空的
                inner_size = int(TILE_SIZE * 0.8)
                offset_y = max(0, (tile.shape[0] - inner_size) // 2) # 處理邊界情況
                offset_x = max(0, (tile.shape[1] - inner_size) // 2) # 處理邊界情況
                inner_tile = tile[offset_y:offset_y + inner_size, offset_x:offset_x + inner_size]
                edges = cv2.Canny(inner_tile, 50, 150)
                edge_density = np.sum(edges) / 255
                if edge_density > 30:  # Adjust threshold for inner tile
                    grid_map[row, col] = 1  # Inaccessible

    # Try to detect exit each frame within the game screen
    exit_coords = find_exit(game_screen, template_path)
    if exit_coords:
        # Adjust exit coordinates based on game border if detected
        if game_border:
            r, c = exit_coords
            last_known_exit = (r + game_border[1] // TILE_SIZE, c + game_border[0] // TILE_SIZE)
        else:
            last_known_exit = exit_coords

    # Use last known if no new one is detected
    if last_known_exit:
        r, c = last_known_exit
        if 0 <= r < grid_rows and 0 <= c < grid_cols: # Ensure within grid bounds
            grid_map[int(r), int(c)] = 2  # Mark exit

    return grid_map

def draw_grid_overlay(screen, grid_map, game_border=None):
    overlay = screen.copy()
    rows, cols = grid_map.shape

    # Calculate offset if game border is detected
    offset_x = game_border[0] if game_border else 0
    offset_y = game_border[1] if game_border else 0

    for row in range(rows):
        for col in range(cols):
            top_left = (int(col * TILE_SIZE + offset_x), int(row * TILE_SIZE + offset_y))
            bottom_right = (int((col + 1) * TILE_SIZE + offset_x), int((row + 1) * TILE_SIZE + offset_y))

            # 確保繪製的矩形在畫面內
            if bottom_right[0] <= screen.shape[1] and bottom_right[1] <= screen.shape[0]:
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
    time.sleep(2)

    while True:
        screen = get_screen()
        game_border = detect_game_border(screen)
        if game_border:
            print(f"Detected game border: {game_border}")
            game_area = crop_to_game_area(screen, game_border)
            grid = get_grid_state(game_area, game_border=game_border)
            grid_overlay = draw_grid_overlay(screen, grid, game_border)
        else:
            grid = get_grid_state(screen)
            grid_overlay = draw_grid_overlay(screen, grid)

        cv2.imshow("Pokémon Gold Grid", grid_overlay)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()