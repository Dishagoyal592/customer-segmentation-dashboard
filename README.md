# 🛍️ Customer Segmentation Dashboard

## 📌 Overview

This project performs customer segmentation using the **K-Means Clustering** algorithm.

Customers are grouped based on their purchasing behaviour to help businesses identify different customer segments and design personalized marketing strategies.

The project is deployed as an interactive **Streamlit web application**.

---

## 🚀 Features

- Interactive Dashboard
- Customer Segmentation using K-Means
- Customer Cluster Prediction
- Business Recommendations
- Interactive Visualizations

---

## 📊 Dataset

**Mall Customers Dataset**

### Features Used

- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

Dataset Size:

- **200 Customers**
- **5 Features**

---

## 🤖 Machine Learning Pipeline

### Data Preprocessing

- Missing value check
- Feature scaling
- Label Encoding (Gender)

### Model Training

**Algorithm Used**

- K-Means Clustering

### Customer Prediction

The application predicts the cluster of a new customer based on:

- Gender
- Age
- Annual Income
- Spending Score

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit

---

## 📂 Project Structure

```text
Customer-Segmentation-Dashboard/
│── app.py
│── Mall_Customers.csv
│── kmeans_model.pkl
│── scaler.pkl
│── requirements.txt
│── README.md
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Live Demo

https://customer-segmentation-dashboard-yjcbvdc3yhkrowerfvnreh.streamlit.app/

---

## 👩‍💻 Author

**Disha Goyal**

B.E. Robotics & AI Engineering

Thapar Institute of Engineering & Technology

---

⭐ If you found this project useful, consider giving it a star!
