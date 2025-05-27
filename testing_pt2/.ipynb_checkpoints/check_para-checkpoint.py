import cv2
import numpy as np
import pyautogui
import time
import os

goal_path = '/Users/violetlin/Documents/github/Pokemon_AI/env_images/cut_tiles/goal'
CHARACTER_IMAGE_FOLDER = '/Users/violetlin/Documents/github/Pokemon_AI/env_images/cut_tiles/character'
TILE_SIZE = 48
last_known_exit = None  # Agent memory
EDGE_THRESHOLD1 = 100
EDGE_THRESHOLD2 = 200
SCALE_STEPS = np.linspace(0.8, 1.2, 5)  # Check scales from 80% to 120%

def get_screen():
    x, y, width, height = 60, 65, 480, 432  # Emulator window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def find_character_by_border(screen, character_image_folder, threshold=0.6):
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen_edges = cv2.Canny(screen_gray, EDGE_THRESHOLD1, EDGE_THRESHOLD2)
    best_match = None
    best_val = -1
    h_best, w_best = 0, 0

    for filename in os.listdir(character_image_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        template_path = os.path.join(character_image_folder, filename)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"⚠️ Warning: Could not read character image {filename}")
            continue

        template_edges = cv2.Canny(template, EDGE_THRESHOLD1, EDGE_THRESHOLD2)

        for scale in SCALE_STEPS:
            resized_template_edges = cv2.resize(template_edges, (0, 0), fx=scale, fy=scale)
            th, tw = resized_template_edges.shape[:2]

            if th > screen_edges.shape[0] or tw > screen_edges.shape[1]:
                continue

            res = cv2.matchTemplate(screen_edges, resized_template_edges, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_val:
                best_val = max_val
                best_match = max_loc
                h_best, w_best = th, tw

    if best_val > threshold and best_match is not None:
        top_left = best_match
        center_x = top_left[0] + w_best // 2
        center_y = top_left[1] + h_best // 2
        col = int(center_x / TILE_SIZE)
        row = int(center_y / TILE_SIZE)
        return (row, col)
    return None

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

def draw_grid_overlay(screen, grid_map, character_position=None):
    overlay = screen.copy()
    rows, cols = grid_map.shape
    for row in range(rows):
        for col in range(cols):
            top_left = (col * TILE_SIZE, row * TILE_SIZE)
            bottom_right = ((col + 1) * TILE_SIZE, (row + 1) * TILE_SIZE)

            color = (255, 255, 255)  # White: walkable
            thickness = 1

            if grid_map[row, col] == 1:
                color = (0, 0, 255)     # Red: inaccessible / border
                thickness = 2
            elif grid_map[row, col] == 2:
                color = (0, 255, 0)     # Green: exit
                thickness = 2

            cv2.rectangle(overlay, top_left, bottom_right, color, thickness)

    if character_position:
        r, c = character_position
        center_x = int((c + 0.5) * TILE_SIZE)
        center_y = int((r + 0.5) * TILE_SIZE)
        cv2.circle(overlay, (center_x, center_y), TILE_SIZE // 3, (0, 255, 0), 2)  # Green circle for character

    return overlay

def save_grid_image(grid_map):
    # Create a blank image to draw the grid
    grid_image = np.zeros((grid_map.shape[0] * TILE_SIZE, grid_map.shape[1] * TILE_SIZE, 3), dtype=np.uint8)

    for row in range(grid_map.shape[0]):
        for col in range(grid_map.shape[1]):
            color = (255, 255, 255)  # Default: white for walkable
            text = ""
            if grid_map[row, col] == 1:
                color = (0, 0, 255)  # Red: inaccessible
                text = "1"  # Inaccessible
            elif grid_map[row, col] == 2:
                color = (0, 255, 0)  # Green: exit
                text = "2"  # Exit
            else:
                text = "0"  # Walkable

            top_left = (col * TILE_SIZE, row * TILE_SIZE)
            bottom_right = ((col + 1) * TILE_SIZE, (row + 1) * TILE_SIZE)
            cv2.rectangle(grid_image, top_left, bottom_right, color, -1)  # Fill the rectangle

            # Add text label for the tile number
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
            text_x = top_left[0] + (TILE_SIZE - text_size[0]) // 2
            text_y = top_left[1] + (TILE_SIZE + text_size[1]) // 2
            cv2.putText(grid_image, text, (text_x, text_y), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)  # Black text

    cv2.imwrite('grid_image_with_numbers.png', grid_image)
    print("Grid image with detection results saved as 'grid_image_with_numbers.png'.")

if __name__ == "__main__":
    print("🎮 Pokémon Grid Detect — Press 'q' to quit, or step on exit.")
    time.sleep(2)

    while True:
        screen = get_screen()
        grid = get_grid_state(screen)
        char_pos = find_character_by_border(screen, CHARACTER_IMAGE_FOLDER)
        grid_overlay = draw_grid_overlay(screen, grid, char_pos)  # Pass character position to drawing function

        # Save the grid image
        save_grid_image(grid)

        cv2.imshow("Pokémon Grid Overlay", grid_overlay)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()