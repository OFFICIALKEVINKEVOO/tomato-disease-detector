# 🍅 Tomato Leaf Disease Detection App

This is a web-based deep learning application that detects diseases in tomato leaves using a custom Convolutional Neural Network (CNN). Built with PyTorch and Streamlit, the app allows users to upload or take a photo of a tomato leaf and receive a prediction with confidence and treatment suggestions.

---

## 🚀 Features

* Upload or take a photo of a tomato leaf.
* Detect 3 classes:

  * Tomato\_healthy
  * Tomato\_Late\_blight
  * Tomato\_Septoria\_leaf\_spot
* Confidence score for predictions.
* Treatment recommendations for detected diseases.
* About tab with project overview and contact info.

---

## 🧠 Model

* Architecture: Custom CNN defined in `model.py`
* Framework: PyTorch
* Training: Done via `train_model.py`
* Evaluations include:

  * Accuracy
  * Confusion matrix (saved in `/models`)
  * Classification report

---

## 📁 Project Structure

```
.
├── app.py                 # Streamlit web app
├── model.py               # CNN model definition
├── train_model.py         # Training script
├── models/
│   ├── model.pth          # Trained model weights
│   └── class_names.json   # Class labels
├── data/                  # Image dataset
│   ├── Tomato_healthy/
│   ├── Tomato_Late_blight/
│   └── Tomato_Septoria_leaf_spot/
├── requirements.txt       # Python dependencies
```

---

## 💻 How to Run Locally

```bash
# 1. Clone the repo
https://github.com/yourusername/tomato-leaf-detector.git

# 2. Navigate to the project folder
cd tomato-leaf-detector

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 🌐 Deployment (Streamlit Cloud)

1. Push this repo to GitHub.
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account.
4. Select this repo and choose `app.py` as the entry point.

✅ Make sure your repo is **public** for free deployment.

---

## 📬 Contact

For questions or suggestions:
**Kelvin Kyanula**
\[kelvinfanuel@gmail.com]
\[OFFICIALKEVINKEVOO]

---

## 📜 License

This project is licensed under the MIT License.
