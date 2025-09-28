# Smart Pause

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/Tkinter-GUI-orange" alt="Tkinter">
  <img src="https://img.shields.io/badge/Memryx-AI%20Accelerator-purple" alt="Memryx">
  <img src="https://img.shields.io/badge/FDLite-Face%20Detection-pink" alt="FDLite">
</p>

**Smart Pause** is a desktop application that helps you stay focused during lectures or study sessions! 
It uses real-time **facial landmark tracking** (via Memryx acceleration + FDLite models) to detect your head orientation and automatically pause/resume playback when you’re distracted.  

*This project was made during MHacks 2025.*


## ✨ Features

- Webcam Tracking  
  Real-time facial landmark detection to monitor head yaw.

- Interactive GUI  
  Built with Tkinter: start, pause, resume, and terminate your session with a click.

- Focus Metrics  
  On session end, view % focus and tracked stats.

- Memryx AI Acceleration  
  Efficient, low-latency inference using Memryx’s `AsyncAccl`.

- Multithreaded Communication
  Socket usage allows for easy GUI interfacing through parent and child threads.

## Frontend Layout

- Live webcam feed with overlay of detected landmarks.
- Control panel with buttons:
  - Start Model → Begin lecture mode.  
  - Pause / Resume → Manually override detection.  
  - End Session → Stop and show statistics.


## Usage

### 1. Clone repo
```bash
git clone https://github.com/wangychn/smart-pause.git
cd smart-pause
```

#### 2. Create environment; ensure open cv is installed
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip wheel

```
#### 3. Install essential packages
```bash
pip install opencv-python==4.11.0.86
pip install pynput==1.8.1
pip install -r requirements.txt
pip3 install --extra-index-url https://developer.memryx.com/pip memryx
```
Note that this project uses Memryx tools; please check out their setup guide for more info!
https://developer.memryx.com/get_started/install_tools.html

#### 4. Run the app! Enjoy!
```bash
python src/python/interface.py
```
