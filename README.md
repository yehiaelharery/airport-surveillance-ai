# 🛫 Airport AI Surveillance System

This project implements a realtime AI-powered surveillance system for airport security.  
It detects **unattended luggage**, **tracks passengers**, and identifies potential **theft scenarios** using modern Computer Vision and ReID models.

---

## 🎯 Main Capabilities

- 🛄 Detect bags and people using **YOLOv8**
- 🎥 Track movement across frames using **BotSORT**
- 🧠 Identify and re-identify passengers using **OSNet**
- ⚠️ Detect unattended or suspicious luggage
- 🚨 Highlight potential theft or unauthorized bag interaction
- 👥 Understand ownership relations between people and bags

---

## Installation & Setup

1. Clone the repository:

```bash
git clone https://github.com/yehiaelharery/airport-surveillance-ai.git
cd airport-surveillance-ai
```

2. (Optional) Create a virtual environment:

```bash
python -m venv venv
# Activate:
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Update paths in `main.py`:

```python
VIDEO_PATH = "path_to_your_video.mp4"
REID_SOURCE_PATH_FOR_TORCHREID = "path_to_osnet_reid_model.pth"
REID_MODEL_FOR_BOXMOT_PATH = "path_to_boxmot_compatible_model.pt"
TRACKING_CONFIG_PATH = "path_to_botsort.yaml"
```

Optional parameters:

- `UNATTENDED_TIME_THRESHOLD` → seconds before a bag is considered unattended  
- `REID_SIMILARITY_THRESHOLD` → cosine similarity threshold for matching people  
- `LUGGAGE_CLASS_IDS` → YOLOv8 class IDs for luggage  

## Usage

Run the system:

```bash
python main.py
```

- Press **q** to quit the live video preview  
- Processed video is saved as `output.mp4`  

## Requirements

- Python 3.8+  
- OpenCV  
- PyTorch  
- TorchReID  
- Ultralytics YOLOv8  
- BoxMOT (BoTSORT / StrongSORT)  

## Future Improvements

- Support live camera feed input  
- Real-time dashboard for statistics  
- Multi-camera / multi-terminal tracking  
- Alerts for theft detection via notifications  

## License

MIT License © [Yehia elharery]
