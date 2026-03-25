# 📡 Deep Learning–Based Wireless Signal Modulation Classifier  
### Using Synthetic IQ Data (AM, FM, BPSK, QPSK, 16-QAM, FSK)

[![Python](https://img.shields.io/badge/Python-3.x-blue)]()
[![MATLAB/Octave](https://img.shields.io/badge/MATLAB/Octave-Signal%20Generation-orange)]()
[![Google Colab](https://img.shields.io/badge/Training-Google%20Colab-green)]()
[![GNU Radio](https://img.shields.io/badge/GNU%20Radio-Optional-yellow)]()
[![DL Model](https://img.shields.io/badge/Model-CNN%2BLSTM-red)]()

---

## 📘 **Project Overview**

This project demonstrates a **deep learning–based modulation classifier** that identifies wireless signal modulation types using **synthetic baseband IQ samples**.

The entire pipeline is designed so you can train and test a modulation classifier **without any SDR hardware**.  
(Though optional GNU Radio support is included for real-world testing.)

### ✔ Modulations Included

| Modulation | Support |
|-----------|---------|
| AM        | ✔ |
| FM        | ✔ |
| BPSK      | ✔ |
| QPSK      | ✔ |
| 16-QAM    | ✔ |
| FSK       | ✔ |

---

## 🌐 **Flow of the Entire Project**

### **1️⃣ IQ Signal Generation (Octave / MATLAB)**  
- Generate clean baseband IQ samples  
- Apply modulation (AM, FM, BPSK, QPSK, QAM, FSK)  
- Normalize signals  
- Export `.mat` / `.npy`

### **2️⃣ Channel Impairments Added**
- AWGN (SNR from −20dB to +20dB)  
- Carrier frequency offset  
- Phase noise  
- Multipath fading (optional)

### **3️⃣ Dataset Preparation**
- Convert signals → shape **(Samples, 2)** for (I, Q)  
- One-hot encode labels  
- Train/test split  
- Save dataset in `dataset/`

### **4️⃣ Deep Learning Model**
- CNN + LSTM hybrid  
- Input: IQ sequence  
- Output: 6 modulation classes  
- Trained in Google Colab GPU  

### **5️⃣ Evaluation**
- Accuracy  
- Loss curves  
- Confusion matrix  
- Accuracy vs SNR  
<img width="822" height="675" alt="image" src="https://github.com/user-attachments/assets/23ceeeb8-d425-4251-8eb9-f42fda0795df" />


### **6️⃣ Optional GNU Radio Testing**
If you have RTL-SDR → test OTA  
If you don’t → use GNU Radio to generate baseband synthetic IQ & evaluate.

---

## 🛠️ Tools Used

### **Required**
- **MATLAB / Octave**
  - Synthetic IQ signal generator
- **Python + TensorFlow/Keras**
  - Model development
- **Google Colab**
  - GPU training environment

### **Optional**
- **GNU Radio (No SDR Required)**
  - Create test waveforms (AM/FM/PSK/QAM)
  - Visualize spectra
  - Export IQ `.bin` for inference

- **RTL-SDR (Optional)**
  - Real-world RF signal capture

---

## 📦 Linux Setup & Installation

### **1️⃣ Install Octave (if no MATLAB)**

```bash
sudo apt update
sudo apt install octave octave-signal octave-communications

```

### **2️⃣ Install Python Dependencies**

```bash
sudo apt install python3 python3-pip
pip install numpy scipy matplotlib tensorflow keras scikit-learn

```

### **3️⃣ Install GNU Radio (Optional)**

```bash
sudo apt install gnuradio

```

### **4️⃣ Clone the Repository**

```bash
git clone https://github.com/gamana29/Deep-Learning-Based-Wireless-Signal-Modulation-Classifier-Using-Synthetic-IQ-Data.git
cd Deep-Learning-Based-Wireless-Signal-Modulation-Classifier-Using-Synthetic-IQ-Data


```
---

### **📁 Project Structure**

```bash
Deep-Learning-Modulation-Classifier/
│
├── colab/
│   └── modulation_training.ipynb
│
├── dataset/
│   └── *.npy   # Generated IQ datasets
│
├── models/
│   └── modulation_cnn.h5
│
├── src/
│   ├── data_generator.py
│   ├── model_cnn.py
│   ├── train.py
│   └── evaluate.py
│
├── gnu_radio/
│   └── modulation_test.grc
│
├── results/
│   └── accuracy_plots.png
│
├── docs/
│
└── README.md

```
---
📊 Model Architecture
---------------------

### **CNN + LSTM Hybrid Network**

This project uses a hybrid deep learning architecture combining **Convolutional Neural Networks (CNNs)** and **Long Short-Term Memory (LSTM)** layers.

#### **Why CNN + LSTM?**
- **CNN layers** extract meaningful temporal features from IQ samples  
- **LSTM layers** learn long-term dependencies such as phase continuity  
- **Dense softmax layer** produces the final modulation class prediction  

#### **Architecture Summary**
- Input shape: **(2048, 2)** → I/Q samples  
- Conv1D → MaxPool  
- Conv1D → MaxPool  
- LSTM layer  
- Dense (ReLU)  
- Dense Softmax (6-class output)

---

🚀 Training in Google Colab
---------------------------

Open the notebook:

## colab/modulation_training.ipynb

### **What the Notebook Does**
✔ Generates synthetic IQ signals (AM, FM, BPSK, QPSK, QAM16, FSK)  
✔ Adds channels: AWGN noise, fading, frequency offset  
✔ Splits data into train/test sets  
✔ Builds CNN + LSTM deep learning model  
✔ Trains for 20 epochs (configurable)  
✔ Saves trained model:  


## models/modulation_cnn.h5
✔ Plots:
- Accuracy & loss curves  
- Confusion matrix  
- Accuracy vs SNR  

---

## **🧪 Model Evaluation**
-------------------

Use:
```bash
python3 src/evaluate.py
```
## **Evaluation includes:**

✔ Accuracy
✔ Loss
✔ Per-class accuracy
✔ Confusion matrix
✔ Accuracy vs SNR

## **Example inference:**

```bash

from tensorflow.keras.models import load_model
import numpy as np

model = load_model("models/modulation_cnn.h5")
iq = np.load("sample.npy")

prediction = model.predict(iq.reshape(1,2048,2))
print("Predicted:", np.argmax(prediction))

```
---

### **GNU Radio Testing(Optional)**
------------------
You can test your trained model with GNU Radio.

**Blocks required:**

-Signal Source -- > Modulator Block (AM/FM/QPSK/QAM) --> Throttle --> Time Sink --> File Sink --> output.bin

## Load .bin file:

```bash
iq = np.fromfile("output.bin", dtype=np.complex64)
iq = np.column_stack((iq.real, iq.imag))
pred = model.predict(iq.reshape(1,2048,2))

```
---

### **📈 Results**
----------------
- 94–98% accuracy (SNR ≥ 0 dB)
- Robust against noise & offsets
- Works on both synthetic & GNU Radio waveforms

---




