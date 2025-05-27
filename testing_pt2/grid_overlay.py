import cv2
import numpy as np
import pyautogui
import time

def get_screen():
    x, y, width, height = 60, 65, 480, 432  # Emulator window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def get_grid_state(screen, tile_size=48):
    height, width, _ = screen.shape
    grid_rows = height // tile_size
    grid_cols = width // tile_size
    grid_map = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    # Convert to edges from the BGR image directly (no grayscale)
    edges = cv2.Canny(screen, 50, 150)

    for row in range(grid_rows):
        for col in range(grid_cols):
            tile = edges[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size]
            edge_density = np.sum(tile) / 255

            if edge_density > 50:  # Adjust as needed
                grid_map[row, col] = 1  # Inaccessible
            else:
                grid_map[row, col] = 0  # Walkable

    # Mark exit (example: top left corner)
    grid_map[0, 0] = 2
    return grid_map

def draw_grid_overlay(screen, grid_map, tile_size=48):
    overlay = screen.copy()
    rows, cols = grid_map.shape
    for row in range(rows):
        for col in range(cols):
            top_left = (col * tile_size, row * tile_size)
            bottom_right = ((col + 1) * tile_size, (row + 1) * tile_size)

            if grid_map[row, col] == 1:
                cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 255), 2)  # Red for blocked
            elif grid_map[row, col] == 2:
                cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 0), 2)  # Green for exit
            else:
                cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), 1)  # White border
    return overlay

if __name__ == "__main__":
    print("Starting live grid capture. Press 'q' to quit.")
    time.sleep(5)  # Give time to focus emulator

    while True:
        screen = get_screen()
        grid_map = get_grid_state(screen)
        grid_overlay = draw_grid_overlay(screen, grid_map)

        cv2.imshow("Pokémon Gold with Grid Overlay", grid_overlay)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
