# 📰 Fake vs Real News Classifier

An end-to-end Machine Learning project that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) techniques. The model is deployed as a user-friendly web application using **Streamlit**.

---

## 🚀 Deployment

🔗 **[Live App on Streamlit Cloud](https://news-classifier-app-f9nwugmadbpgjw6rk7dk4d.streamlit.app/)**

---

## 📌 Project Overview

With the rise of misinformation on the internet, particularly through social media platforms, identifying fake news has become a critical challenge. This project develops a **scalable, intelligent NLP-based classifier** to detect whether a news article is **genuine** or **fraudulent**.

---

## 🧠 Key Features

- Cleaned and preprocessed real-world news data from Kaggle  
- Feature extraction using **TF-IDF**  
- Model training using **Logistic Regression** (92% Accuracy)  
- Evaluation using F1-score, accuracy, and confusion matrix  
- **Streamlit-based Web UI** for interactive predictions  
- Real-time prediction with **confidence scores**

---

## 🛠️ Tools & Technologies

| Category           | Tools/Libs Used                           |
|--------------------|--------------------------------------------|
| Language           | Python 3                                   |
| Libraries          | `Scikit-learn`, `NLTK`, `Pandas`, `NumPy`, `Joblib` |
| Text Processing    | NLTK (stopword removal, tokenization)      |
| Vectorization      | TF-IDF (Top 5000 features)                 |
| ML Algorithms      | Logistic Regression, Naive Bayes           |
| Deployment         | Streamlit Cloud, GitHub                    |
| Visualization      | Matplotlib (local)                         |
| Dataset Source     | [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) |

---

## 📂 Project Structure

```bash
├── app.py                   # Streamlit App Script
├── model.pkl                # Trained Logistic Regression Model
├── tfidf_vectorizer.pkl     # Saved TF-IDF Vectorizer
├── requirements.txt         # Required Python libraries
├── README.md                # Project Overview (this file)
├── data/
│   ├── True.csv             # Real news dataset
│   └── Fake.csv             # Fake news dataset
```

---

## 📈 Model Development Workflow

1. **Data Collection**: Combined `True.csv` and `Fake.csv`, added binary labels (1 = Real, 0 = Fake).
2. **Preprocessing**: Tokenization, lowercasing, punctuation & stopword removal using NLTK.
3. **Feature Engineering**: TF-IDF vectorization, limited to 5000 top features.
4. **Modeling**: Trained and compared Logistic Regression and Naive Bayes.
5. **Evaluation**: Measured with Accuracy (~92%), F1-score, and confusion matrix.
6. **Web UI**: Developed Streamlit frontend for clean and responsive predictions.
7. **Deployment**: Hosted via GitHub and Streamlit Cloud.

---

## ✅ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/news-fake-real-classifier.git
cd news-fake-real-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 👩‍💻 Author

**Uribindi Dharshini**  
🔗 [LinkedIn](https://www.linkedin.com/in/dharshini-u-44a5412a4)