# 🌿 Plant Disease Detection using Deep Learning

> 🌾 A deep learning approach for early detection of plant diseases to assist farmers and improve agricultural productivity.

---

## 📘 Overview

This project focuses on **automatic detection of plant diseases** from leaf images using **Convolutional Neural Networks (CNNs)** and **Transfer Learning**.

The goal is to assist farmers and agricultural experts in **early disease diagnosis**, reducing crop damage and increasing productivity.

📓 The notebook [`Projet_plant_disease_.ipynb`](Projet_plant_disease_.ipynb) contains the **entire workflow** — from data preprocessing to model training, evaluation, and prediction.

---

## 🎯 Objectives

- ✅ Design and train a deep learning model to identify multiple plant diseases with high accuracy  
- 🔁 Experiment with **custom CNNs** and **pre-trained architectures** (VGG16, ResNet50, MobileNetV2)  
- 📊 Perform comparative analysis using **Accuracy, Precision, Recall, and F1-Score**  
- 🚀 Demonstrate model deployment readiness  

---

## 🧠 Technologies & Tools

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Programming** | Python |
| **Deep Learning** | TensorFlow, Keras |
| **Data Processing** | NumPy, Pandas, OpenCV |
| **Visualization** | Matplotlib, Seaborn |
| **Development** | Google Colab |
| **Evaluation** | Scikit-learn metrics |

---

## 🗂️ Dataset

The dataset contains labeled images of **healthy** and **diseased plant leaves**.  
Each image corresponds to a plant species and a disease category.

📦 **Dataset Used:** [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

**Structure Example:**

data/
├── train/
│ ├── Apple___Black_rot/
│ ├── Apple___healthy/
│ └── ...
├── validation/
└── test/

markdown
Copier le code

- 🌱 **Total Classes:** 38  
- 🖼️ **Total Images:** ~54,000  
- 📏 **Image Size:** 224×224 pixels  

---

## ⚙️ Preprocessing Pipeline

- 🧾 **Data Loading:** Images labeled from folder structure  
- 📐 **Image Resizing:** All images resized to 224×224  
- 🎚️ **Normalization:** Pixel values scaled to [0, 1]  
- 🌈 **Augmentation:** Real-time augmentation using `ImageDataGenerator`  
  - Rotation  
  - Zoom  
  - Flip (horizontal/vertical)  
  - Brightness shift  
- 📊 **Split:** Train (80%) / Validation (10%) / Test (10%)

---

## 🏗️ Model Architectures

### 🧩 1. Custom CNN (Baseline)

python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
Optimizer: Adam

Loss: CategoricalCrossentropy

Learning Rate: 0.001

🧠 2. Transfer Learning Models
Model	Base Architecture	Frozen Layers	Trainable Layers	Input Size
VGG16	ImageNet	15	5	224×224
ResNet50	ImageNet	140	20	224×224
MobileNetV2	ImageNet	120	30	224×224

Custom classification head:

python
Copier le code
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
🧪 Training & Evaluation
📦 Batch Size: 32

⏱️ Epochs: 25

⚙️ Optimizer: Adam

🧭 Scheduler: ReduceLROnPlateau

⛑️ Callbacks: EarlyStopping, ModelCheckpoint

Metrics Evaluated:

Accuracy

Precision

Recall

F1-Score

📊 Results Summary
Model	Accuracy	Precision	Recall	F1-Score	Training Time
Custom CNN	88.7%	0.88	0.87	0.87	~25 min
VGG16 (TL)	94.8%	0.94	0.94	0.94	~30 min
ResNet50 (TL) 🏆	96.2%	0.96	0.96	0.96	~35 min
MobileNetV2 (TL)	95.5%	0.95	0.95	0.95	~27 min

✅ Best Model: ResNet50 (fine-tuned) – 96.2% accuracy, excellent generalization.

🔍 Visualization Examples
📈 Confusion Matrix

🧩 Training Curves

🔥 Grad-CAM Heatmaps

Example:

python
Copier le code
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
🔮 Future Work
🌐 Deploy as a Streamlit web app

📱 Mobile-based inference with TensorFlow Lite

🧠 Integrate Explainable AI (Grad-CAM, SHAP)

🧩 Experiment with Vision Transformers (ViT)

⚡ How to Run Locally
bash
Copier le code
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook Projet_plant_disease_.ipynb
Test a new leaf image:

python
Copier le code
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('best_model.h5')
img = image.load_img('sample_leaf.jpg', target_size=(224,224))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = np.argmax(model.predict(img_array))
print("Predicted class:", class_names[prediction])
📦 Dependencies
shell
Copier le code
tensorflow>=2.9
keras>=2.9
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
👨‍💻 Author
Zakaria Dahbi
🎓 Master’s in Computer Science & Artificial Intelligence — Université Ibn Tofail
💡 Passionate about AI, Machine Learning & Intelligent Systems
🔗 LinkedIn Profile

📜 License
This project is licensed under the MIT License.
You are free to use, modify, and distribute this code with proper attribution.
