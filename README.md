
⭐ Deep Learning–Based Wireless Signal Modulation Classifier
Using Synthetic IQ Data (AM, FM, BPSK, QPSK, QAM, FSK)

This project demonstrates a deep learning–based modulation classifier using synthetic baseband IQ signals, trained in Google Colab, and optionally testable using GNU Radio Companion.

The goal is to classify signals such as:

Modulation	Included?
AM	✔
FM	✔
BPSK	✔
QPSK	✔
16-QAM	✔
FSK	✔

The model is trained on synthetically generated IQ data, allowing training without SDR hardware.

📁 Project Structure
Deep-Learning-Based-Wireless-Signal-Modulation-Classifier/
│
├── colab/                # Google Colab training notebooks
│   └── modulation_training.ipynb
│
├── dataset/              # Generated IQ datasets (.npy)
│
├── models/               # Saved models (.h5, .tflite, .onnx)
│   └── modulation_cnn.h5
│
├── src/                  # Python source code for training/testing
│   ├── data_generator.py
│   ├── model_cnn.py
│   ├── train.py
│   └── evaluate.py
│
├── gnu_radio/            # GNU Radio flowgraphs
│   └── modulation_test.grc
│
├── results/              # Accuracy plots, confusion matrix
│
├── docs/                 # Documentation
│
├── README.md
└── .gitignore

🚀 Features
✔ Synthetic IQ Signal Generator

Generates AM, FM, PSK, QAM, FSK

Adds AWGN noise with configurable SNR

Frequency/phase offsets

Multipath fading (optional)

✔ Deep Learning Classifier

CNN + LSTM hybrid architecture

Input: (I, Q) samples

Output: modulation class label

✔ Evaluation

Test accuracy

Confusion matrix

Accuracy vs. SNR curves

✔ GNU Radio Integration

Load signals into GRC

Visualize spectrogram/FFT

Simulate wireless channel (no SDR needed)

🧠 Model File

This repo includes:

models/modulation_cnn.h5


This saved model can be loaded in Python:

from tensorflow.keras.models import load_model
model = load_model("models/modulation_cnn.h5")

🧪 Testing With GNU Radio (NO SDR NEEDED)

You can generate test waveforms using:

Required Blocks:

✔ Signal Source
✔ Modulate (AM/FM/PSK/QAM)
✔ Throttle
✔ QT GUI Time Sink
✔ File Sink (export .bin IQ)

Then test in Python:

import numpy as np
iq = np.fromfile("gnu_radio/output.bin", dtype=np.complex64)
pred = model.predict(iq.reshape(1, -1, 2))

📌 How to Train in Google Colab

Open:

colab/modulation_training.ipynb


Run all cells to:

✔ Generate dataset
✔ Train model
✔ Save model
✔ Plot results

📈 Example Results

CNN Accuracy: 94–98% (SNR ≥ 0 dB)

Robust to noise & frequency offset

Fast real-time inference

🧑‍💻 Author

Gamana
Deep Learning & Wireless Signal Processing Research

⭐ How to Cite
Gamana (2025). Deep Learning–Based Wireless Signal Modulation Classifier Using Synthetic IQ Data.
GitHub: https://github.com/gamana29/Deep-Learning-Based-Wireless-Signal-Modulation-Classifier-Using-Synthetic-IQ-Data

🚀 Future Work

Add RNN/Transformer model

Real OTA dataset (RTL-SDR/PlutoSDR)

Deploy on mobile/edge TPUs

🙌 Contributions Welcome!

Feel free to submit PRs or raise issues.


👏 Star the Repo If You Found It Useful!
✅ 4. Add Files to GitHub

Now run:

git add .
git commit -m "Added project structure and README"
git push -u origin main

