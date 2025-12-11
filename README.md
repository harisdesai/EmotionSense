# EmotionSense
Real-Time Emotion Detection using Face (CNN) + Voice (SVM/ML)  
**Author:** Sohail Badakar (Student)  
**Project:** 3rd Year AI/ML Academic Project

## Overview
EmotionSense combines facial expression recognition and voice-tone analysis to classify emotions (happy, sad, angry, neutral, surprise, fear, disgust). It contains training scripts, a real-time demo, and a Streamlit UI.

## Repository Structure
- `face_model_train.py` - Train CNN on FER2013.
- `voice_model_train.py` - Extract MFCC features and train SVM for RAVDESS.
- `realtime_demo.py` - Real-time webcam + (optional) mic-based demo using trained models.
- `app.py` - Streamlit demo to upload image/audio and show predictions.
- `utils.py` - Helper functions (feature extraction, preprocessing).
- `requirements.txt` - Python dependencies.
- `report.md` - Academic report (convert to PDF for submission).
- `dataset_notes.md` - Links and notes for datasets.
- `LICENSE` - MIT

## Quick start
1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate    # windows
pip install -r requirements.txt
```
2. Prepare datasets (see `dataset_notes.md`) and place them in `datasets/`.
3. Train face model:
```bash
python face_model_train.py
```
4. Train voice model:
```bash
python voice_model_train.py
```
5. Run real-time demo:
```bash
python realtime_demo.py
```
6. Streamlit UI:
```bash
streamlit run app.py
```

## Notes
- Models and datasets are **not** included (size). Use links in `dataset_notes.md`.
- For submission, convert `report.md` to PDF (many editors or `pandoc` can do it).

