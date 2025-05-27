import cv2
import numpy as np
import pyautogui
import time
import os

goal_path = '/Users/violetlin/Documents/github/Pokemon_AI/env_images/cut_tiles/goal'
CHARACTER_IMAGE_FOLDER = '/Users/violetlin/Documents/github/Pokemon_AI/env_images/cut_tiles/character'  # <--- Ensure this is correct
TILE_SIZE = 48
last_known_exit = None  # Agent memory
EDGE_THRESHOLD1 = 100
EDGE_THRESHOLD2 = 200
SCALE_STEPS = np.linspace(0.8, 1.2, 5) # Check scales from 80% to 120%
FLOOR_HSV_LOW = np.array([3,24,141])   # HSV 下限 94,24,141
FLOOR_HSV_HIGH = np.array([26,109,223]) # HSV 上限 117
 
def get_screen():
    x, y, width, height = 60, 65, 480, 432  # Emulator window region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def find_character_by_border(screen, character_image_folder, threshold=0.6):
    """Detects the character based on its border using edge detection."""
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

    # # 確保輸入為24bit (3通道) 格式
    # if len(screen.shape) == 2 or screen.shape[2] == 1:
    #     screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2BGR)
    
    # 從外計算縮小一圈 (邊緣去除TILE_SIZE範圍)
    margin = TILE_SIZE
    h, w = screen.shape[:2]
    cropped = screen[margin:h-margin, margin:w-margin] if (h > 2*margin and w > 2*margin) else screen.copy()

    # 讀取模板並調整大小
    template = cv2.imread(template_path, 1)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    if template is None:
        print("❌ Exit template not found!")
        return None
    template = cv2.resize(template, (TILE_SIZE, TILE_SIZE))

    # 多尺度模板匹配
    best_match = None
    best_val = -1
    for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:  # 擴展尺度範圍
        resized_template = cv2.resize(template, (int(TILE_SIZE*scale), int(TILE_SIZE*scale)))
        if cropped.shape[0] < resized_template.shape[0] or cropped.shape[1] < resized_template.shape[1]:
            continue
        
        # 預處理 (高斯模糊)
        screen_blur = cv2.GaussianBlur(cropped, (5, 5), 0)
        template_blur = cv2.GaussianBlur(resized_template, (5, 5), 0)
        
        # 使用改進的匹配方法
        res = cv2.matchTemplate(screen_blur, template_blur, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_val:
            best_val = max_val
            best_match = max_loc

    # 判斷匹配結果
    if best_val > 0.72:  # 調整閾值
        match_x = best_match[0] + margin
        match_y = best_match[1] + margin
        
        # 轉換為grid座標
        col = match_x // TILE_SIZE
        row = match_y // TILE_SIZE
        
        # 確保座標在合理範圍內
        if 0 <= row < (h // TILE_SIZE) and 0 <= col < (w // TILE_SIZE):
            last_known_exit = (row, col)
            return (row, col)
    
    return None

def get_grid_state(screen, template_path=f'{goal_path}/downstairs.png'):
    global last_known_exit
    height, width, _ = screen.shape
    grid_rows = height // TILE_SIZE
    grid_cols = width // TILE_SIZE
    grid_map = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

    # 使用高斯模糊進行預處理
    blurred = cv2.GaussianBlur(screen, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 地板顏色檢測
    floor_mask = cv2.inRange(hsv, FLOOR_HSV_LOW, FLOOR_HSV_HIGH)
    floor_mask = cv2.GaussianBlur(floor_mask, (3, 3), 0)

    # 邊界顏色檢測
    gray_mask = cv2.inRange(hsv, (0, 0, 50), (180, 60, 180))
    gray_mask = cv2.GaussianBlur(gray_mask, (3, 3), 0)

    for row in range(grid_rows):
        for col in range(grid_cols):
            # 檢查地板區域
            floor_tile = floor_mask[row*TILE_SIZE:(row+1)*TILE_SIZE, col*TILE_SIZE:(col+1)*TILE_SIZE]
            floor_ratio = np.sum(floor_tile > 0) / (TILE_SIZE * TILE_SIZE)
            
            # 檢查邊界區域
            gray_tile = gray_mask[row*TILE_SIZE:(row+1)*TILE_SIZE, col*TILE_SIZE:(col+1)*TILE_SIZE]
            gray_density = np.sum(gray_tile) / 255
            
            if gray_density > TILE_SIZE * TILE_SIZE * 0.2:
                grid_map[row, col] = 1  # 邊界/不可通行 (紅色)
            elif floor_ratio >= 0.8:    # 使用設定的threshold
                grid_map[row, col] = 0  # 地板區域 (白色/可通行)
            else:
                grid_map[row, col] = 3  # 其他區域 (藍色)

    # 出口檢測
    exit_coords = find_exit(screen, template_path)
    if not exit_coords and last_known_exit:
        print("🔁 Using last known exit.")
        exit_coords = last_known_exit

    if exit_coords:
        r, c = exit_coords
        grid_map[r, c] = 2  # 標記出口 (綠色)

    return grid_map

def draw_grid_overlay(screen, grid_map, character_position=None):
    overlay = screen.copy()
    rows, cols = grid_map.shape
    
    # 繪製所有網格
    for row in range(rows):
        for col in range(cols):
            top_left = (col * TILE_SIZE, row * TILE_SIZE)
            bottom_right = ((col + 1) * TILE_SIZE, (row + 1) * TILE_SIZE)

            # 根據網格狀態設定顏色
            if grid_map[row, col] == 0:    # 可通行區域
                color = (255, 255, 255)   # 白色
                thickness = 1
            elif grid_map[row, col] == 1:  # 不可通行邊界
                color = (0, 0, 255)       # 紅色
                thickness = 2
            elif grid_map[row, col] == 2:  # 出口
                color = (0, 255, 0)       # 綠色
                thickness = 3
                # 強化出口標記
                cv2.putText(overlay, "EXIT", (top_left[0] + 5, top_left[1] + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif grid_map[row, col] == 3:  # 不確定區域
                color = (255, 0, 0)       # 藍色
                thickness = 2

            cv2.rectangle(overlay, top_left, bottom_right, color, thickness)

    # 標記角色位置 (優先繪製在最上層)
    if character_position:
        r, c = character_position
        center_x = int((c + 0.5) * TILE_SIZE)
        center_y = int((r + 0.5) * TILE_SIZE)
        cv2.circle(overlay, (center_x, center_y), TILE_SIZE // 3, (0, 255, 0), -1)  # 實心圓
        cv2.putText(overlay, "YOU", (center_x - 15, center_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return overlay

if __name__ == "__main__":
    print("🎮 Pokémon Grid Detect — Press 'q' to quit, or step on exit.")
    time.sleep(2)

    while True:
        screen = get_screen()
        
        # 確保輸入為24bit
        if len(screen.shape) == 2:
            screen = cv2.cvtColor(screen, cv2.COLOR_GRAY2BGR)
        
        grid = get_grid_state(screen)
        char_pos = find_character_by_border(screen, CHARACTER_IMAGE_FOLDER)
        grid_overlay = draw_grid_overlay(screen, grid, char_pos)

        cv2.imshow("Pokémon Grid Overlay", grid_overlay)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()



# def find_exit(screen, template_path=f'{goal_path}/downstairs.png'):
#     global TILE_SIZE, last_known_exit

#     template = cv2.imread(template_path)
#     if template is None:
#         print("❌ Exit template not found!")
#         return None

#     template = cv2.resize(template, (TILE_SIZE, TILE_SIZE))
#     margin = TILE_SIZE
#     cropped = screen[margin:-margin, margin:-margin]

#     best_match = None
#     best_val = -1
#     for scale in [0.96, 0.98, 1.0, 1.02, 1.04]:
#         resized_template = cv2.resize(template, (int(TILE_SIZE*scale), int(TILE_SIZE*scale)))
#         if cropped.shape[0] < resized_template.shape[0] or cropped.shape[1] < resized_template.shape[1]:
#             continue
#         screen_blur = cv2.GaussianBlur(cropped, (3, 3), 0)
#         res = cv2.matchTemplate(screen_blur, resized_template, cv2.TM_CCOEFF_NORMED)
#         _, max_val, _, max_loc = cv2.minMaxLoc(res)
#         if max_val > best_val:
#             best_val = max_val
#             best_match = max_loc

#     if best_val > 0.74:  # lowered threshold a bit
#         match_x = best_match[0] + margin
#         match_y = best_match[1] + margin
#         col = round(match_x / TILE_SIZE)
#         row = round(match_y / TILE_SIZE)
#         last_known_exit = (row, col)
#         return (row, col)

#     return None

# def get_grid_state(screen, template_path=f'{goal_path}/downstairs.png'):
#     global last_known_exit
#     height, width, _ = screen.shape
#     grid_rows = height // TILE_SIZE
#     grid_cols = width // TILE_SIZE
#     grid_map = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

#     # 使用您喜歡的參數進行預處理
#     blurred = cv2.GaussianBlur(screen, (5, 5), 0)
#     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
#     # 地板顏色檢測 (調整為您的參數)
#     floor_mask = cv2.inRange(hsv, FLOOR_HSV_LOW, FLOOR_HSV_HIGH)  # 請先定義 FLOOR_HSV_LOW/HIGH
#     floor_mask = cv2.GaussianBlur(floor_mask, (3, 3), 0)

#     # 邊界顏色檢測 (保留原有灰色檢測)
#     gray_mask = cv2.inRange(hsv, (0, 0, 50), (180, 60, 180))
#     gray_mask = cv2.GaussianBlur(gray_mask, (3, 3), 0)

#     for row in range(grid_rows):
#         for col in range(grid_cols):
#             # 檢查地板區域 (使用您喜歡的 threshold=0.8)
#             floor_tile = floor_mask[row*TILE_SIZE:(row+1)*TILE_SIZE, col*TILE_SIZE:(col+1)*TILE_SIZE]
#             floor_ratio = np.sum(floor_tile > 0) / (TILE_SIZE * TILE_SIZE)
            
#             # 檢查邊界區域
#             gray_tile = gray_mask[row*TILE_SIZE:(row+1)*TILE_SIZE, col*TILE_SIZE:(col+1)*TILE_SIZE]
#             gray_density = np.sum(gray_tile) / 255
            
#             if gray_density > TILE_SIZE * TILE_SIZE * 0.2:
#                 grid_map[row, col] = 1  # 邊界/不可通行 (紅色)
#             elif floor_ratio >= 0.8:    # 使用您設定的 threshold
#                 grid_map[row, col] = 0  # 地板區域 (白色/可通行)
#             else:
#                 grid_map[row, col] = 3  # 其他區域 (新增分類)

#     # 出口檢測 (保留原有邏輯)
#     exit_coords = find_exit(screen, template_path)
#     if not exit_coords and last_known_exit:
#         print("🔁 Using last known exit.")
#         exit_coords = last_known_exit

#     if exit_coords:
#         r, c = exit_coords
#         grid_map[r, c] = 2  # 標記出口 (綠色)

#     return grid_map

# def draw_grid_overlay(screen, grid_map, character_position=None):
#     overlay = screen.copy()
#     rows, cols = grid_map.shape
#     for row in range(rows):
#         for col in range(cols):
#             top_left = (col * TILE_SIZE, row * TILE_SIZE)
#             bottom_right = ((col + 1) * TILE_SIZE, (row + 1) * TILE_SIZE)

#             # 顏色設定 (整合您的參數)
#             if grid_map[row, col] == 0:    # 地板 (可通行)
#                 color = (255, 255, 255)   # 白色
#                 thickness = 1
#             elif grid_map[row, col] == 1:  # 邊界 (不可通行)
#                 color = (0, 0, 255)       # 紅色
#                 thickness = 2
#             elif grid_map[row, col] == 2:  # 出口
#                 color = (0, 255, 0)       # 綠色
#                 thickness = 2
#             elif grid_map[row, col] == 3:  # 新增: 不確定區域
#                 color = (255, 0, 0)       # 藍色 (您原本的地板顏色)
#                 thickness = 2

#             cv2.rectangle(overlay, top_left, bottom_right, color, thickness)

#     # 角色標記 (保留原有邏輯)
#     if character_position:
#         r, c = character_position
#         center_x = int((c + 0.5) * TILE_SIZE)
#         center_y = int((r + 0.5) * TILE_SIZE)
#         cv2.circle(overlay, (center_x, center_y), TILE_SIZE // 3, (0, 255, 0), 2)

#     return overlay

# if __name__ == "__main__":
#     print("🎮 Pokémon Grid Detect — Press 'q' to quit, or step on exit.")
#     time.sleep(2)

#     while True:
#         screen = get_screen()
#         grid = get_grid_state(screen)
#         char_pos = find_character_by_border(screen, CHARACTER_IMAGE_FOLDER)
#         grid_overlay = draw_grid_overlay(screen, grid, char_pos) # Pass character position to drawing function

#         cv2.imshow("Pokémon Grid Overlay", grid_overlay)

#         if cv2.waitKey(100) & 0xFF == ord('q'):
#             break

#     cv2.destroyAllWindows()