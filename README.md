# 📰 Fake News Detection using Machine Learning  

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Samar-111/Fake-News-Detection/blob/main/Fake_News_Detection.ipynb)  

---

## 📌 Project Overview  
This project builds a *Fake News Classifier* using *Natural Language Processing (NLP)* and *Machine Learning*.  
The model predicts whether a news article is *Real (0)* or *Fake (1)* based on its text content.  

We use *TF-IDF Vectorization* for feature extraction and train multiple ML models:  
- Logistic Regression  
- Naive Bayes  
- Random Forest  

---

## 📂 Dataset  
We use the *Fake News Dataset* from [Kaggle Fake News Detection](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets).  

- Columns: id, title, author, text, label  
- Target: label (0 = Real, 1 = Fake)  

---

## ⚙ Tech Stack  
- *Python 3*  
- *Jupyter Notebook / Google Colab*  
- *Libraries:*  
  - pandas, numpy  
  - scikit-learn  
  - matplotlib, seaborn  
  - re, string  

---

## 🚀 How to Run  

### ▶ Run Online (Recommended)  
Click the badge above ☝ to open in *Google Colab*.  

### 💻 Run Locally  
1. Clone this repository:  
   ```bash
   git clone https://github.com/Samar-111/Fake-News-Detection.git
   cd Fake-News-Detection
2.Install dependencies:
   pip install -r requirements.txt
3.Run the Jupyter Notebook:
 
jupyter notebook Fake_News_Detection.ipynb

📊 Results

Logistic Regression: Best performance with high accuracy.

Naive Bayes: Fast and decent accuracy.

Random Forest: Works but less effective for sparse text.


✅ Evaluations included:

Accuracy, Precision, Recall, F1-Score

Confusion Matrix (Heatmap)

ROC Curve & AUC Score



---

🔮 Future Improvements

Try LSTMs / GRU / Transformers (BERT) for deep learning-based detection.

Deploy as a Flask/Streamlit web app.

Add WordCloud visualization for Fake vs Real articles.



---

📌 Project Structure

📁 Fake-News-Detection
│── 📄 Fake_News_Detection.ipynb   # Main Notebook
│── 📄 train.csv                   # Dataset (from Kaggle)
│── 📄 requirements.txt            # Dependencies
│── 📄 README.md                   # Project Documentation

👨‍💻 Author

👤Samar Anand
🔗[GitHub](https://github.com/Samar-111) 
