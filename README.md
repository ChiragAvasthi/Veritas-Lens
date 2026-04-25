# 🎥 Deepfake Video Detection System

## 📌 Overview
With the rapid rise of manipulated media, detecting deepfake videos has become increasingly important.  
This project focuses on identifying whether a video is real or fake using deep learning techniques.  
The system analyzes video frames and detects inconsistencies to classify the content.

---

## 🚀 Features
- Detects deepfake videos using AI-based models  
- Extracts and analyzes frames from input videos  
- Provides prediction results based on learned patterns  
- Simple and easy-to-run implementation  

---

## 🛠 Tech Stack
- Python
- Streamlit
- OpenCV  
- TensorFlow / PyTorch  
- NumPy, Pandas  

---

## 📂 Project Structure
  deepfake-video-detection/
│── train.py
│── utils.py
│── requirements.txt
│── README.md

---

---

## ▶️ How to Run

1. Clone the repository:

git clone https://github.com/talkwithharsh/deepfake-video-detection.git

cd deepfake-video-detection


2. Install dependencies:

pip install -r requirements.txt


3. Run the project:

python train.py


---

## 📦 Model
The trained model is not uploaded due to large file size.  
You can download it from here: https://drive.google.com/drive/folders/1P-Ze64c1JMZTvg9ve0w5G8mUMgEkrIDN?usp=sharing

---


# Deepfake Video Detection System

A production-ready Streamlit web app that detects deepfake videos using a **ResNeXt50_32x4d + LSTM** model.

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **dlib** needs a C++ compiler and cmake:
> - Ubuntu : `sudo apt-get install cmake build-essential`
> - macOS  : `brew install cmake`
> - Windows: install [CMake](https://cmake.org/download/) + Visual Studio Build Tools

---

### 2 — Download models

```bash
python download_models.py
```

This fetches all `.pt` files from the shared Google Drive folder into `models/`.  
Alternatively, copy your `.pt` files manually into the `models/` folder.

---

### 3 — Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
deepfake_app/
├── app.py                # Streamlit UI
├── model.py              # ResNeXt50 + LSTM architecture
├── utils.py              # Full inference pipeline
├── download_models.py    # One-click model downloader
├── requirements.txt
├── README.md
│
├── models/               # .pt model files go here
│   ├── df_model_1.pt
│   └── df_model_2.pt
│
└── temp/                 # Auto-created; temporary video files
```

---

## Inference Pipeline

```
Video upload
    ↓
Extract 150 frames  (uniform sampling)
    ↓
Face detection      (dlib frontal face detector)
    ↓
Crop → Resize 112×112 → Normalize (ImageNet)
    ↓
Tensor  (1, 150, 3, 112, 112)
    ↓
ResNeXt50  feature extraction
    ↓
LSTM       temporal modelling
    ↓
Softmax  →  REAL / FAKE + Confidence %
```

---

## Model Architecture

| Component        | Detail                            |
|------------------|-----------------------------------|
| Backbone         | ResNeXt50_32x4d (no FC/avgpool)   |
| Pooling          | AdaptiveAvgPool2d(1,1)            |
| Temporal model   | LSTM  hidden = 2048               |
| Dropout          | 0.4                               |
| Classifier       | Linear(2048 → 2)                  |
| Input shape      | `(1, 150, 3, 112, 112)`           |
| Classes          | 0 = REAL · 1 = FAKE               |

---

## Key Remapping (automatic)

Models trained with different attribute names are **automatically remapped** on load:

| Saved key prefix | Mapped to              |
|------------------|------------------------|
| `model.*`        | `feature_extractor.*`  |
| `linear1.*`      | `classifier.*`         |
| `module.*`       | *(DataParallel strip)*  |

No manual conversion needed — drop any compatible `.pt` into `models/` and it works.

---

## Error Reference

| Message | Cause | Fix |
|---------|-------|-----|
| "Video only has N frames" | Clip too short | Use video ≥ 150 frames |
| "Only N usable faces detected" | Face not visible throughout | Use clearer face video |
| Model Error / weight mismatch | Wrong checkpoint format | Use a ResNeXt+LSTM `.pt` |
| No models in dropdown | `models/` folder empty | Run `download_models.py` |

---


## 👨‍💻 Author
Harsh Kumar

GPU (CUDA) is used automatically if available. CPU fallback is fully supported.
