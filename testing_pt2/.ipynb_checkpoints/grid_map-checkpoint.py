import cv2
import numpy as np
import pyautogui
import time
import os

goal_path = '/Users/violetlin/Documents/github/Pokemon_AI/env_images/cut_tiles/goal'
EXIT_TEMPLATE_PATH = f'{goal_path}/downstairs.png' 

TILE_SIZE = 48 
FLOOR_HSV_LOW = np.array([3,24,141])   # HSV 下限
FLOOR_HSV_HIGH = np.array([26,109,223]) # HSV 上限

def get_screen():
    x, y, width, height = 60, 65, 480, 432  # 模擬器視窗區域
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def get_all_pixels_in_grid(screen):
    height, width = screen.shape[:2]
    grid_cols = width // TILE_SIZE
    grid_rows = height // TILE_SIZE
    grid_pixels = np.zeros((grid_rows, grid_cols, TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)

    for i in range(grid_rows):
        for j in range(grid_cols):
            x_start = j * TILE_SIZE
            x_end = (j + 1) * TILE_SIZE
            y_start = i * TILE_SIZE
            y_end = (i + 1) * TILE_SIZE
            grid_pixels[i, j] = screen[y_start:y_end, x_start:x_end]
    
    return grid_pixels

def get_grid_state(screen, template_path=EXIT_TEMPLATE_PATH):
    height, width = screen.shape[:2]
    grid_rows = height // TILE_SIZE
    grid_cols = width // TILE_SIZE
    grid_map = np.zeros((grid_rows, grid_cols), dtype=np.uint8)
    all_pixels = get_all_pixels_in_grid(screen)
    
    # 檢測地板
    for i in range(grid_rows):
        for j in range(grid_cols):
            grid_pixels = all_pixels[i, j]
            hsv_grid = cv2.cvtColor(grid_pixels, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_grid, FLOOR_HSV_LOW, FLOOR_HSV_HIGH)
            match_ratio = np.sum(mask > 0) / (TILE_SIZE * TILE_SIZE)
            grid_map[i, j] = 1 if match_ratio >= 0.8 else 0
    
    # 檢測出口
    exit_pos = detect_exit_tile(screen, all_pixels, template_path)
    if exit_pos:
        i, j = exit_pos
        grid_map[i, j] = 2
    
    return grid_map, all_pixels, exit_pos

def get_dominant_colors(img, k=3, center_ratio=0.8):
    """獲取圖像中心區域的主導顏色 (k-means 聚類)"""
    h, w = img.shape[:2]
    center_size = int(min(h, w) * center_ratio)
    cx, cy = w // 2, h // 2
    half = center_size // 2
    
    # 安全裁剪中心區域
    patch = img[max(cy-half,0):min(cy+half,h), max(cx-half,0):min(cx+half,w)]
    
    # 轉換到 HSV 顏色空間 (只取 H 通道)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0].reshape(-1, 1).astype(np.float32)
    
    # K-means 聚類找出主色
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(hue, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 返回排序後的主色 (按出現頻率)
    unique, counts = np.unique(labels, return_counts=True)
    sorted_colors = centers[unique[np.argsort(-counts)]].flatten()
    return sorted_colors[:k]  # 返回前 k 個主色

def detect_exit_tile(image, all_pixels, EXIT_TEMPLATE_PATH, color_threshold=15):
    """純顏色特徵比對版"""
    # 讀取並預處理模板
    template = cv2.imread(EXIT_TEMPLATE_PATH)
    if template is None:
        print(f"Error: Cannot load template at {EXIT_TEMPLATE_PATH}")
        return None
    
    # 強力雜訊過濾 (雙邊濾波保留邊緣 + 高斯模糊)
    template_clean = cv2.bilateralFilter(template, 9, 75, 75)
    template_clean = cv2.GaussianBlur(template_clean, (3, 3), 1)
    
    # 獲取模板主色 (HSV 色調通道)
    template_colors = get_dominant_colors(template_clean, k=3)
    
    exit_coord = None
    min_color_diff = float('inf')
    grid_rows, grid_cols = all_pixels.shape[:2]
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            grid = all_pixels[i, j]
            
            # 相同預處理流程
            grid_clean = cv2.bilateralFilter(grid, 9, 75, 75)
            grid_clean = cv2.GaussianBlur(grid_clean, (3, 3), 1)
            
            # 獲取網格主色
            grid_colors = get_dominant_colors(grid_clean, k=3)
            
            # 計算主色差異 (取前3個主色的平均差異)
            color_diff = np.mean(np.abs(template_colors - grid_colors[:len(template_colors)]))
            
            if color_diff < min_color_diff and color_diff < color_threshold:
                min_color_diff = color_diff
                exit_coord = (i, j)
    
    return exit_coord if min_color_diff < color_threshold else None

## L2 norm
# def get_sorted_color_vector(img, center_size=48):
#     h, w = img.shape[:2]
#     cx, cy = w // 2, h // 2
#     half = center_size // 2

#     center_patch = img[cy - half:cy + half, cx - half:cx + half]
#     pixels = center_patch.reshape(-1, 3)
#     sorted_pixels = np.sort(pixels, axis=0)
#     return sorted_pixels.flatten()

# def detect_exit_tile(image, all_pixels, EXIT_TEMPLATE_PATH):
#     # 讀取模板圖片
#     template = cv2.imread(EXIT_TEMPLATE_PATH)
#     if template is None:
#         print("Error: Exit template not found!")
#         return None
    
#     # 獲取網格單元格的尺寸
#     tile_height, tile_width = all_pixels[0,0].shape[:2]

#     # 將模板圖片調整為與網格單元格相同大小
#     resized_template = cv2.resize(template, (tile_width, tile_height))
    
#     # 對模板圖片進行模糊處理
#     # template_blur = cv2.GaussianBlur(resized_template, (5, 5), 0)
#     template_vector = get_sorted_color_vector(resized_template, center_size=30)

#     min_diff = float('inf')
#     exit_coord = None
#     grid_rows, grid_cols = all_pixels.shape[:2]

#     for i in range(grid_rows):
#         for j in range(grid_cols):
#             grid = all_pixels[i, j]
#             grid = cv2.resize(grid, (tile_width, tile_height))
#             # blur_grid = cv2.GaussianBlur(grid, (5, 5), 0)
#             grid_vector = get_sorted_color_vector(grid, center_size=30)

#             # 計算向量之間的差異
#             diff = np.linalg.norm(template_vector - grid_vector)
#             if diff < min_diff:
#                 min_diff = diff
#                 exit_coord = (i, j)

#     return exit_coord

def draw_grid_overlay(screen, grid_map, all_pixels, exit_pos=None):
    overlay = screen.copy()
    grid_rows, grid_cols = grid_map.shape
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            x1, y1 = j * TILE_SIZE, i * TILE_SIZE
            x2, y2 = x1 + TILE_SIZE, y1 + TILE_SIZE
            
            if exit_pos == (i, j):
                color = (0, 255, 0)  # 綠色框（出口）
            elif grid_map[i, j] == 1:
                color = (255, 0, 0)  # 藍色框（地板）
            else:
                color = (0, 0, 255) # 紅色框（非地板）
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    
    return overlay

def main():
    print("🎮 Pokémon Grid Detect — Press 'q' to quit")
    time.sleep(2)
    
    while True:
        screen = get_screen()
        grid_map, all_pixels, exit_pos = get_grid_state(screen)
        overlay = draw_grid_overlay(screen, grid_map, all_pixels, exit_pos)
        
        cv2.imshow("Pokémon Grid Overlay", overlay)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()