import cv2
import numpy as np
import pyautogui
import time
import os

# --- Configuration ---
# TODO: Ensure these paths are correct for your system
goal_path = '/Users/violetlin/Documents/github/Pokemon_AI/env_images/cut_tiles/goal'
CHARACTER_IMAGE_FOLDER = '/Users/violetlin/Documents/github/Pokemon_AI/env_images/cut_tiles/character'

TILE_SIZE = 48
last_known_exit = None  # Agent memory for exit location (stores both grid and pixel coordinates)
last_known_exit_template = None  # Stores the template image that was matched

# Canny Edge Detection parameters (primarily for character detection)
EDGE_THRESHOLD1 = 100
EDGE_THRESHOLD2 = 200
SCALE_STEPS = np.linspace(0.8, 1.2, 5) # Character scale check

# HSV Color ranges (for floor/border detection in get_grid_state)
# TODO: Tune these HSV values based on your specific game visuals
FLOOR_HSV_LOW = np.array([3, 24, 141])   # Example HSV lower bound for floor
FLOOR_HSV_HIGH = np.array([26, 109, 223]) # Example HSV upper bound for floor
GRAY_BORDER_HSV_LOW = np.array([0, 0, 50])      # Example HSV lower bound for gray borders
GRAY_BORDER_HSV_HIGH = np.array([180, 60, 180]) # Example HSV upper bound for gray borders

# Exit Detection Parameters
EXIT_TEMPLATE_FILENAME = 'downstairs.png' # Make sure this file exists in goal_path
EXIT_SEARCH_MARGIN = TILE_SIZE // 2 # Reduced margin to potentially find exits closer to edge
EXIT_SCALE_RANGE = [0.95, 1.0, 1.05] # Slightly tighter scale range, assuming less variation
EXIT_MATCH_THRESHOLD = 0.65 # Slightly lower threshold, adjust based on testing
EXIT_BLUR_KERNEL = (3, 3)    # Smaller blur kernel size
EXIT_CONFIRMATION_FRAMES = 3 # Number of consecutive frames exit must be detected to confirm
exit_confirmation_count = 0  # Counter for exit confirmation

DEBUG_SHOW_MATCH = False # Set to True to visualize the template match area


def get_screen():
    """Captures the specified emulator window region."""
    # TODO: Verify these coordinates match your emulator window precisely
    x, y, width, height = 60, 65, 480, 432  # Emulator window region
    try:
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        # Convert PIL Image to NumPy array (RGB) and then to BGR for OpenCV
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"❌ Error capturing screen: {e}")
        return None

# [Rest of your existing functions remain the same until find_exit]

# --- Updated Exit Detection ---
def find_exit(screen, template_path):
    """
    Finds the exit using color template matching.
    Now stores both grid and pixel coordinates of the exit when found.
    Includes confirmation logic to prevent false positives.
    """
    global TILE_SIZE, last_known_exit, last_known_exit_template, EXIT_SEARCH_MARGIN
    global EXIT_SCALE_RANGE, EXIT_MATCH_THRESHOLD, EXIT_BLUR_KERNEL, DEBUG_SHOW_MATCH
    global exit_confirmation_count, EXIT_CONFIRMATION_FRAMES

    if screen is None:
        print("❌ Screen capture failed, cannot find exit.")
        return None

    # 1. Load the template (ensure it's a clean image of the exit tile)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR) # Load as color (BGR)
    if template is None:
        print(f"❌ Exit template not found or could not be read: {template_path}")
        return None

    # Optional: Resize template *once* if it's not already TILE_SIZE x TILE_SIZE
    template = cv2.resize(template, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)
    th_base, tw_base = template.shape[:2]

    # 2. Define Search Region (crop screen slightly to reduce search area)
    h_screen, w_screen = screen.shape[:2]
    margin_y = min(EXIT_SEARCH_MARGIN, h_screen // 2 -1)
    margin_x = min(EXIT_SEARCH_MARGIN, w_screen // 2 -1)

    if h_screen <= 2 * margin_y or w_screen <= 2 * margin_x:
         print("⚠️ Screen too small for specified margin, searching whole screen.")
         cropped_screen = screen.copy()
         offset_x, offset_y = 0, 0
    else:
        cropped_screen = screen[margin_y : h_screen - margin_y, margin_x : w_screen - margin_x]
        offset_x, offset_y = margin_x, margin_y # Remember the offset for final coordinates

    if cropped_screen.shape[0] < th_base or cropped_screen.shape[1] < tw_base:
         print("⚠️ Cropped screen is smaller than base template, cannot match.")
         return None

    # 3. Preprocessing (Apply slight blur to reduce noise sensitivity)
    screen_blur = cv2.GaussianBlur(cropped_screen, EXIT_BLUR_KERNEL, 0)
    template_blur = template # Using non-blurred template assuming it's clean

    # 4. Multi-Scale Template Matching
    best_match_loc = None
    best_match_val = -1
    best_scale_used = 1.0
    best_template_dims = (th_base, tw_base)

    for scale in EXIT_SCALE_RANGE:
        new_w = int(tw_base * scale)
        new_h = int(th_base * scale)

        if new_w <= 0 or new_h <= 0: continue

        resized_template = cv2.resize(template_blur, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        th, tw = resized_template.shape[:2]

        if th > screen_blur.shape[0] or tw > screen_blur.shape[1]:
            continue

        try:
            res = cv2.matchTemplate(screen_blur, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_match_val:
                best_match_val = max_val
                best_match_loc = max_loc
                best_scale_used = scale
                best_template_dims = (th, tw)

        except cv2.error as e:
            continue

    # --- Debugging ---
    if DEBUG_SHOW_MATCH and best_match_loc is not None:
        debug_screen = cropped_screen.copy()
        match_end_pt = (best_match_loc[0] + best_template_dims[1], best_match_loc[1] + best_template_dims[0])
        cv2.rectangle(debug_screen, best_match_loc, match_end_pt, (255, 0, 255), 2)
        cv2.imshow("Exit Match Debug", debug_screen)
    # ---------------

    # 5. Evaluate Match and Determine Coordinates
    if best_match_val >= EXIT_MATCH_THRESHOLD and best_match_loc is not None:
        # Found a potential match - increment confirmation counter
        exit_confirmation_count += 1
        
        if exit_confirmation_count >= EXIT_CONFIRMATION_FRAMES:
            # Confirmed exit detection
            match_x_local = best_match_loc[0]
            match_y_local = best_match_loc[1]

            # Add the offset back to get coordinates relative to the original screen
            match_x_global = match_x_local + offset_x
            match_y_global = match_y_local + offset_y

            # Calculate the center of the matched region
            center_x = match_x_global + best_template_dims[1] // 2
            center_y = match_y_global + best_template_dims[0] // 2

            # Convert center coordinates to grid cell coordinates
            col = int(center_x // TILE_SIZE)
            row = int(center_y // TILE_SIZE)

            # Store both pixel and grid coordinates
            last_known_exit = {
                'grid': (row, col),
                'pixel': (center_x, center_y),
                'template': template.copy(),  # Store the template that was matched
                'scale': best_scale_used,
                'confidence': best_match_val
            }
            
            print(f"✅ Exit CONFIRMED at grid ({row}, {col}) pixel ({center_x}, {center_y}) with score {best_match_val:.3f} scale {best_scale_used:.2f}")
            return (row, col)
        else:
            # Not enough confirmations yet
            print(f"⚠️ Potential exit found (score: {best_match_val:.3f}), need {EXIT_CONFIRMATION_FRAMES - exit_confirmation_count} more confirmations")
            return None
    else:
        # No match found this frame - reset confirmation counter
        exit_confirmation_count = 0
        return None

# [Rest of your existing functions remain the same]

def get_grid_state(screen, exit_template_path):
    """Generates the grid map based on color detection and exit finding."""
    global last_known_exit, TILE_SIZE, FLOOR_HSV_LOW, FLOOR_HSV_HIGH, GRAY_BORDER_HSV_LOW, GRAY_BORDER_HSV_HIGH

    if screen is None:
         print("❌ Cannot get grid state, screen capture failed.")
         return None

    height, width, _ = screen.shape
    grid_rows = height // TILE_SIZE
    grid_cols = width // TILE_SIZE
    grid_map = np.full((grid_rows, grid_cols), 3, dtype=np.uint8)

    # Preprocessing for color detection
    blurred = cv2.GaussianBlur(screen, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Color masks
    floor_mask = cv2.inRange(hsv, FLOOR_HSV_LOW, FLOOR_HSV_HIGH)
    gray_mask = cv2.inRange(hsv, GRAY_BORDER_HSV_LOW, GRAY_BORDER_HSV_HIGH)

    # Analyze each tile
    for row in range(grid_rows):
        for col in range(grid_cols):
            y1, y2 = row * TILE_SIZE, (row + 1) * TILE_SIZE
            x1, x2 = col * TILE_SIZE, (col + 1) * TILE_SIZE

            floor_tile = floor_mask[y1:y2, x1:x2]
            gray_tile = gray_mask[y1:y2, x1:x2]

            total_pixels = TILE_SIZE * TILE_SIZE
            if total_pixels == 0: continue

            floor_ratio = np.count_nonzero(floor_tile) / total_pixels
            gray_density = np.count_nonzero(gray_tile) / total_pixels

            if gray_density > 0.3:
                grid_map[row, col] = 1
            elif floor_ratio > 0.7:
                grid_map[row, col] = 0

    # --- Exit Detection ---
    exit_coords = find_exit(screen, exit_template_path)

    if exit_coords:
        # Successfully detected exit THIS frame
        r, c = exit_coords
        if 0 <= r < grid_rows and 0 <= c < grid_cols:
             grid_map[r, c] = 2
        else:
             print(f"⚠️ Detected exit coords ({r},{c}) out of grid bounds ({grid_rows}x{grid_cols}).")
             exit_coords = None

    # If no exit detected this frame but we have a last known exit
    if not exit_coords and last_known_exit:
        # Verify the last known exit is still valid by checking the stored template
        r_last, c_last = last_known_exit['grid']
        center_x, center_y = last_known_exit['pixel']
        
        # Extract the region where we expect the exit to be
        half_size = TILE_SIZE // 2
        x1 = max(0, int(center_x - half_size))
        y1 = max(0, int(center_y - half_size))
        x2 = min(width, int(center_x + half_size))
        y2 = min(height, int(center_y + half_size))
        
        if x2 > x1 and y2 > y1:  # Ensure valid region
            exit_region = screen[y1:y2, x1:x2]
            
            # Resize template to match the scale it was originally found at
            template = last_known_exit['template']
            scaled_template = cv2.resize(template, (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
            
            # Check if the template still matches the region
            res = cv2.matchTemplate(exit_region, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            
            if max_val > EXIT_MATCH_THRESHOLD * 0.8:  # Slightly lower threshold for verification
                if 0 <= r_last < grid_rows and 0 <= c_last < grid_cols:
                    grid_map[r_last, c_last] = 2
                    print(f"🔁 Using memorized exit at ({r_last},{c_last}) with confidence {max_val:.3f}")
            else:
                print(f"⚠️ Memorized exit no longer matches (score: {max_val:.3f}). Forgetting.")
                last_known_exit = None

    return grid_map

# [Rest of your main loop remains the same]
if __name__ == "__main__":
    print("--- Pokémon Grid Detector ---")
    print(f"Tile Size: {TILE_SIZE}x{TILE_SIZE}")
    print(f"Exit Template: {os.path.join(goal_path, EXIT_TEMPLATE_FILENAME)}")
    print(f"Exit Threshold: {EXIT_MATCH_THRESHOLD}")
    print("Press 'q' in the 'Pokémon Grid Overlay' window to quit.")
    if DEBUG_SHOW_MATCH:
        print("DEBUG_SHOW_MATCH is True: Showing exit match visualization.")
    print("-----------------------------")
    time.sleep(2)

    # Construct the full path to the exit template
    exit_template_full_path = os.path.join(goal_path, EXIT_TEMPLATE_FILENAME)
    if not os.path.exists(exit_template_full_path):
         print(f"🚨 FATAL ERROR: Exit template not found at {exit_template_full_path}")
         print("Please check the 'goal_path' and 'EXIT_TEMPLATE_FILENAME' variables.")
         exit() # Stop execution if template is missing

    if not os.path.isdir(CHARACTER_IMAGE_FOLDER):
         print(f"🚨 WARNING: Character image folder not found at {CHARACTER_IMAGE_FOLDER}")
         print("Character detection will not work.")


    while True:
        # 1. Get Screen
        screen = get_screen()
        if screen is None:
             print("Screen capture failed, skipping frame.")
             time.sleep(1) # Wait a bit before retrying
             continue

        # Ensure screen is BGR format for consistency
        if len(screen.shape) == 2: # Grayscale
            screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2BGR)
        elif screen.shape[2] == 4: # BGRA (e.g., from some capture methods)
            screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)


        # 2. Detect Character (Optional, might be unreliable)
        char_pos = find_character_by_border(screen, CHARACTER_IMAGE_FOLDER)
        # if char_pos: print(f"Character found at grid: {char_pos}")
        # else: print("Character not found.")


        # 3. Generate Grid State (includes exit detection)
        grid = get_grid_state(screen, exit_template_full_path)
        if grid is None:
             print("Failed to generate grid state, skipping frame.")
             time.sleep(0.5)
             continue

        # 4. Draw Overlay
        grid_overlay = draw_grid_overlay(screen, grid, char_pos)

        # 5. Display
        cv2.imshow("Pokémon Grid Overlay", grid_overlay)

        # 6. Check for Quit Key
        key = cv2.waitKey(100) & 0xFF # Check keys every 100ms
        if key == ord('q'):
            print("'q' pressed, exiting...")
            break
        # Optional: Add a key to toggle debug view
        elif key == ord('d'):
             DEBUG_SHOW_MATCH = not DEBUG_SHOW_MATCH
             print(f"DEBUG_SHOW_MATCH set to: {DEBUG_SHOW_MATCH}")
             if not DEBUG_SHOW_MATCH and cv2.getWindowProperty("Exit Match Debug", 0) >= 0:
                  cv2.destroyWindow("Exit Match Debug")


    # Cleanup
    cv2.destroyAllWindows()
    print("Application finished.")