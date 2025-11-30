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

## 🧠 Tech Stack

| Component | Library / Model |
|----------|------------------|
| Object Detection | YOLOv8 |
| Tracking | BotSORT |
| Re-Identification | OSNet |
| Code | Python |
| Engine | OpenCV, Torch |

---

## ▶️ How to Run

```bash
git clone https://github.com/yehiaelharery/airport-surveillance-ai
cd airport-surveillance-ai
pip install -r requirements.txt
python main.py
