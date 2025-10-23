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
last_known_exit = None  # Agent memory for exit location

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

# --- Character Detection (Keep as is, but note it might be unreliable) ---
def find_character_by_border(screen, character_image_folder, threshold=0.6):
    """Detects the character based on its border using edge detection."""
    if screen is None: return None
    # Check if screen is already grayscale
    if len(screen.shape) == 3 and screen.shape[2] == 3:
         screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    elif len(screen.shape) == 2:
         screen_gray = screen # Already grayscale
    else:
         print("⚠️ Unsupported screen format for character detection")
         return None

    screen_edges = cv2.Canny(screen_gray, EDGE_THRESHOLD1, EDGE_THRESHOLD2)
    best_match_loc = None
    best_match_val = -1
    best_match_scale_dims = (0, 0)

    if not os.path.isdir(character_image_folder):
        print(f"❌ Character image folder not found: {character_image_folder}")
        return None

    for filename in os.listdir(character_image_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        template_path = os.path.join(character_image_folder, filename)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"⚠️ Warning: Could not read character image {filename}")
            continue

        template_edges = cv2.Canny(template, EDGE_THRESHOLD1, EDGE_THRESHOLD2)
        if template_edges.shape[0] == 0 or template_edges.shape[1] == 0:
             print(f"⚠️ Warning: Empty edges for character template {filename}")
             continue


        for scale in SCALE_STEPS:
            # Calculate new dimensions
            new_w = int(template_edges.shape[1] * scale)
            new_h = int(template_edges.shape[0] * scale)

            # Ensure dimensions are positive
            if new_w <= 0 or new_h <= 0:
                continue

            # Resize template edges
            resized_template_edges = cv2.resize(template_edges, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            th, tw = resized_template_edges.shape[:2]

            # Check if template fits within screen edges
            if th > screen_edges.shape[0] or tw > screen_edges.shape[1]:
                continue # Skip if scaled template is larger than screen

            try:
                res = cv2.matchTemplate(screen_edges, resized_template_edges, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > best_match_val:
                    best_match_val = max_val
                    best_match_loc = max_loc
                    best_match_scale_dims = (th, tw) # Store height, width of best scaled template
            except cv2.error as e:
                 # This can happen if the template is larger than the image after cropping/scaling
                 # print(f"-> OpenCV error during character matchTemplate (scale {scale:.2f}): {e}")
                 continue # Skip this scale


    if best_match_val > threshold and best_match_loc is not None:
        top_left = best_match_loc
        # Use the dimensions of the template *at the scale it was matched*
        h_best, w_best = best_match_scale_dims
        center_x = top_left[0] + w_best // 2
        center_y = top_left[1] + h_best // 2

        # Convert center pixel coordinates to grid coordinates
        col = int(center_x // TILE_SIZE)
        row = int(center_y // TILE_SIZE)
        return (row, col) # Return grid coordinates

    return None # No character found above threshold


# --- Updated Exit Detection ---
def find_exit(screen, template_path):
    """
    Finds the exit using color template matching.
    More robust to minor variations and includes debugging visualization.
    """
    global TILE_SIZE, last_known_exit, EXIT_SEARCH_MARGIN, EXIT_SCALE_RANGE, EXIT_MATCH_THRESHOLD, EXIT_BLUR_KERNEL, DEBUG_SHOW_MATCH

    if screen is None:
        print("❌ Screen capture failed, cannot find exit.")
        return None

    # 1. Load the template (ensure it's a clean image of the exit tile)
    template = cv2.imread(template_path, cv2.IMREAD_COLOR) # Load as color (BGR)
    if template is None:
        print(f"❌ Exit template not found or could not be read: {template_path}")
        return None

    # Optional: Resize template *once* if it's not already TILE_SIZE x TILE_SIZE
    # If your template is already perfectly sized, you can comment this out.
    template = cv2.resize(template, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)
    th_base, tw_base = template.shape[:2]


    # 2. Define Search Region (crop screen slightly to reduce search area)
    h_screen, w_screen = screen.shape[:2]
    # Ensure margin doesn't make crop dimensions negative or zero
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
    # Optional: Blur template too if it has noise, otherwise keep it sharp
    # template_blur = cv2.GaussianBlur(template, EXIT_BLUR_KERNEL, 0)
    template_blur = template # Using non-blurred template assuming it's clean


    # 4. Multi-Scale Template Matching
    best_match_loc = None
    best_match_val = -1 # Initialize with a value lower than any possible score
    best_scale_used = 1.0
    best_template_dims = (th_base, tw_base)

    for scale in EXIT_SCALE_RANGE:
        # Calculate new dimensions based on base template size
        new_w = int(tw_base * scale)
        new_h = int(th_base * scale)

        if new_w <= 0 or new_h <= 0: continue # Skip invalid scales

        resized_template = cv2.resize(template_blur, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        th, tw = resized_template.shape[:2]

        # Check if scaled template fits within the cropped screen
        if th > screen_blur.shape[0] or tw > screen_blur.shape[1]:
            # print(f"-> Scaled template (scale {scale:.2f}) too large for cropped screen.")
            continue # Skip if scaled template is larger than the area we search in

        try:
            # Perform template matching
            res = cv2.matchTemplate(screen_blur, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            # Update best match if current match is better
            if max_val > best_match_val:
                best_match_val = max_val
                best_match_loc = max_loc
                best_scale_used = scale
                best_template_dims = (th, tw) # Store dims of the matched template

        except cv2.error as e:
            # print(f"-> OpenCV error during exit matchTemplate (scale {scale:.2f}): {e}")
            continue # Skip this scale

    # --- Debugging ---
    # print(f"Best exit match value: {best_match_val:.4f} (Threshold: {EXIT_MATCH_THRESHOLD})")
    if DEBUG_SHOW_MATCH and best_match_loc is not None:
        debug_screen = cropped_screen.copy()
        match_end_pt = (best_match_loc[0] + best_template_dims[1], best_match_loc[1] + best_template_dims[0])
        cv2.rectangle(debug_screen, best_match_loc, match_end_pt, (255, 0, 255), 2) # Magenta box
        cv2.imshow("Exit Match Debug", debug_screen)
    # ---------------

    # 5. Evaluate Match and Determine Coordinates
    if best_match_val >= EXIT_MATCH_THRESHOLD and best_match_loc is not None:
        # Found a good match
        match_x_local = best_match_loc[0] # X in cropped coordinates
        match_y_local = best_match_loc[1] # Y in cropped coordinates

        # Add the offset back to get coordinates relative to the original screen
        match_x_global = match_x_local + offset_x
        match_y_global = match_y_local + offset_y

        # Calculate the center of the matched region
        center_x = match_x_global + best_template_dims[1] // 2
        center_y = match_y_global + best_template_dims[0] // 2

        # Convert center coordinates to grid cell coordinates
        col = int(center_x // TILE_SIZE)
        row = int(center_y // TILE_SIZE)

        # Ensure coordinates are within the grid bounds
        grid_rows_total = h_screen // TILE_SIZE
        grid_cols_total = w_screen // TILE_SIZE
        if 0 <= row < grid_rows_total and 0 <= col < grid_cols_total:
            # print(f"✅ Exit FOUND at grid ({row}, {col}) with score {best_match_val:.3f} scale {best_scale_used:.2f}")
            last_known_exit = (row, col) # Update memory
            return (row, col)
        else:
            print(f"⚠️ Exit match found, but grid coords ({row}, {col}) are out of bounds.")
            return None # Match found but calculated coords are invalid
    else:
        # No confident match found
        # print(f"❌ Exit NOT found (Best score: {best_match_val:.3f})")
        return None # Explicitly return None if no exit found


# --- Grid State Generation ---
def get_grid_state(screen, exit_template_path):
    """Generates the grid map based on color detection and exit finding."""
    global last_known_exit, TILE_SIZE, FLOOR_HSV_LOW, FLOOR_HSV_HIGH, GRAY_BORDER_HSV_LOW, GRAY_BORDER_HSV_HIGH

    if screen is None:
         print("❌ Cannot get grid state, screen capture failed.")
         return None # Cannot proceed without screen

    height, width, _ = screen.shape
    grid_rows = height // TILE_SIZE
    grid_cols = width // TILE_SIZE
    # Initialize grid: 0=Walkable(Floor), 1=Border/Wall, 3=Other/Unknown
    grid_map = np.full((grid_rows, grid_cols), 3, dtype=np.uint8) # Default to Unknown

    # Preprocessing for color detection
    blurred = cv2.GaussianBlur(screen, (3, 3), 0) # Use small blur
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Color masks
    floor_mask = cv2.inRange(hsv, FLOOR_HSV_LOW, FLOOR_HSV_HIGH)
    gray_mask = cv2.inRange(hsv, GRAY_BORDER_HSV_LOW, GRAY_BORDER_HSV_HIGH)

    # Analyze each tile
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Calculate tile boundaries
            y1, y2 = row * TILE_SIZE, (row + 1) * TILE_SIZE
            x1, x2 = col * TILE_SIZE, (col + 1) * TILE_SIZE

            # Extract tile regions from masks
            floor_tile = floor_mask[y1:y2, x1:x2]
            gray_tile = gray_mask[y1:y2, x1:x2]

            # Calculate ratios/densities
            # Use np.count_nonzero for potentially faster calculation
            total_pixels = TILE_SIZE * TILE_SIZE
            if total_pixels == 0: continue # Avoid division by zero

            floor_ratio = np.count_nonzero(floor_tile) / total_pixels
            gray_density = np.count_nonzero(gray_tile) / total_pixels # Density is ratio for binary mask

            # Classify tile (prioritize borders)
            # TODO: Adjust these thresholds based on observation
            if gray_density > 0.3: # If >30% of tile matches border color range
                grid_map[row, col] = 1  # Border / Inaccessible (Red)
            elif floor_ratio > 0.7: # If >70% of tile matches floor color range
                grid_map[row, col] = 0  # Floor / Walkable (White)
            # Else: remains 3 (Unknown/Other - Blue)


    # --- Exit Detection ---
    exit_coords = find_exit(screen, exit_template_path)

    if exit_coords:
        # Successfully detected exit THIS frame
        r, c = exit_coords
        # Ensure coords are valid before marking map
        if 0 <= r < grid_rows and 0 <= c < grid_cols:
             grid_map[r, c] = 2  # Mark exit (Green)
        else:
             print(f"⚠️ Detected exit coords ({r},{c}) out of grid bounds ({grid_rows}x{grid_cols}).")
             # Decide if you want to use last_known_exit even if detection gave OOB coords
             exit_coords = None # Treat as not found if out of bounds

    if not exit_coords and last_known_exit:
        # Exit not detected this frame, use memory if available
        # print("🔁 Exit not found, using last known location.")
        r_last, c_last = last_known_exit
        # Ensure last known coords are still valid before marking map
        if 0 <= r_last < grid_rows and 0 <= c_last < grid_cols:
             grid_map[r_last, c_last] = 2 # Mark last known exit (Green)
        else:
             print(f"⚠️ Last known exit coords ({r_last},{c_last}) are out of grid bounds ({grid_rows}x{grid_cols}). Forgetting.")
             last_known_exit = None # Invalidate memory if out of bounds

    return grid_map


# --- Drawing Overlay ---
def draw_grid_overlay(screen, grid_map, character_position=None):
    """Draws the grid and optional character position onto the screen."""
    if screen is None or grid_map is None:
         print("⚠️ Cannot draw overlay, missing screen or grid map.")
         # Return a black screen or the original screen if available
         return screen if screen is not None else np.zeros((432, 480, 3), dtype=np.uint8)

    overlay = screen.copy()
    rows, cols = grid_map.shape

    # Define colors BGR
    color_walkable = (255, 255, 255) # White
    color_border = (0, 0, 255)     # Red
    color_exit = (0, 255, 0)       # Green
    color_unknown = (255, 0, 0)     # Blue
    color_char = (0, 255, 255)    # Yellow

    for row in range(rows):
        for col in range(cols):
            top_left = (col * TILE_SIZE, row * TILE_SIZE)
            bottom_right = ((col + 1) * TILE_SIZE, (row + 1) * TILE_SIZE)

            cell_type = grid_map[row, col]
            color = color_unknown # Default
            thickness = 1

            if cell_type == 0: # Walkable
                color = color_walkable
            elif cell_type == 1: # Border
                color = color_border
                thickness = 2
            elif cell_type == 2: # Exit
                color = color_exit
                thickness = 3
                # Optional: Add text label to exit tile
                cv2.putText(overlay, "EXIT", (top_left[0] + 3, bottom_right[1] - 5),
                            cv2.FONT_HERSHEY_PLAIN, 0.8, color_exit, 1)
            # elif cell_type == 3: # Unknown (already default)
            #     color = color_unknown
            #     thickness = 1 # Thin lines for unknown

            cv2.rectangle(overlay, top_left, bottom_right, color, thickness)

    # Draw character position if found
    if character_position:
        r, c = character_position
        # Ensure character coords are valid before drawing
        if 0 <= r < rows and 0 <= c < cols:
             center_x = int((c + 0.5) * TILE_SIZE)
             center_y = int((r + 0.5) * TILE_SIZE)
             radius = TILE_SIZE // 4
             cv2.circle(overlay, (center_x, center_y), radius, color_char, -1) # Filled yellow circle
             # Optional: Add outline or text
             # cv2.circle(overlay, (center_x, center_y), radius, (0,0,0), 1) # Black outline

    return overlay


# --- Main Loop ---
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