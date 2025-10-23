# 🎮 AI Plays Pokémon Gold via Visual Detection

This project uses computer vision and reinforcement learning to enable an AI agent to play **Pokémon Gold** on a Gameboy emulator. The agent identifies the game map visually, overlays a grid, and navigates using Q-learning.

---

## 🚀 Project Overview

- **Goal**: Build an AI agent that can visually interpret the game screen and autonomously navigate the map.
- **Platform**: Gameboy emulator (e.g., VisualBoyAdvance)
- **Game**: Pokémon Gold
- **Main Method**: Grid-based map recognition + Q-learning navigation

---

## 🧠 Core Workflow

### 01. Realtime Screen Capture
- Captures the emulator window region using `pyautogui`.
- Enables frame-by-frame analysis of the game screen.

### 02. Draw Grid
- Overlays a grid on the captured screen.
- Each cell represents a navigable tile in the game world.

### 03. Environment Recognition
- Detects terrain and exits using color segmentation and image matching.
- Generates a grid-based map of the current environment.

### 04. Navigation
- Implements **Q-learning** to train the agent to move intelligently.
- Agent learns optimal paths based on rewards and penalties.

---

## 🛠 Dependencies

- `pyautogui` – for screen capture and input simulation  
- `opencv-python` – for image processing  
- `numpy` – for grid and Q-table operations  
- `matplotlib` – for debugging and visualization  
- Gameboy emulator (e.g., VisualBoyAdvance)

---

## 📦 Setup Instructions

1. Download and install a Gameboy emulator.
2. Load Pokémon Gold ROM.
3. Run the Python script to start screen capture and grid overlay.
4. Train the agent using Q-learning or load a pre-trained model.

---

## 📈 Future Improvements

- Add support for dynamic map changes (e.g., indoors vs outdoors)
- Integrate OCR for dialog and menu recognition
- Expand agent behavior to include item collection and battle decisions

---

## 🧑‍💻 Author

YuChien Lin — AI tinkerer, visual prompt engineer, and Pokémon enthusiast.


