# 🛡️ Face Mask Detection Using Real-Time Video Frames with Deep Learning

This is a real-time face mask detection system built using deep learning and computer vision. It classifies face mask usage into four distinct categories:

- 😷 N95 Mask
- 🩺 Surgical Mask
- ⚠️ Improperly Worn Mask
- ❌ No Mask

This project was developed by **Dharaneesh .V** during the internship program *“Advanced Applications of AI and Machine Learning”* conducted from **May 15 to June 15, 2025**.

---

## 🔧 Features

- Real-time webcam-based mask detection
- Four-class classification using MobileNetV2
- Lightweight and fast — runs on CPU systems
- Easy integration and open-source

---

## 🧰 Tech Stack

- **Language**: Python 3.8+
- **Framework**: TensorFlow / Keras
- **Libraries**: OpenCV, NumPy, imutils
- **Model**: MobileNetV2 (Transfer Learning)

---

## 📁 Dataset

The dataset includes 1,000+ labeled images for each of the four classes. Data augmentation techniques such as:
- Rotation
- Flipping
- Brightness scaling
- Zooming  
were applied to improve accuracy and generalization.

---

## 🧠 Model Workflow

1. **Face Detection** using OpenCV DNN
2. **Preprocessing** (resize, normalize, encode)
3. **Prediction** using MobileNetV2
4. **Labeling** the video stream with results

---

## 📷 Output Example

Bounding boxes are drawn around detected faces with the predicted label and confidence score. Colors represent:
- 🟢 N95 Mask
- 🔵 Surgical Mask
- 🟡 Improper Mask
- 🔴 No Mask


##IMPORTANT 

- In the project file named mask_detector.model.h5 is pretrained model file
- And in the project file named plot.png is the pretrained model graph
- So train the model with the dataset the above will be created automatically.Just delete the above two files when you download the project file.
