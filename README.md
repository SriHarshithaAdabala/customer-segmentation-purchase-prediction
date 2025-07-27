# 🛒 Customer Segmentation & Purchase Prediction (ML-based Streamlit App)
<img width="1911" height="754" alt="image" src="https://github.com/user-attachments/assets/06071742-8a2a-4630-bd73-785094782918" />
An interactive Streamlit application...

## 📸 App Preview

### 🔹 Input & Prediction Panel
<img width="1909" height="760" alt="image" src="https://github.com/user-attachments/assets/3310b0d2-1360-449d-87bd-ef88e50d0e92" />
<img width="1918" height="756" alt="image" src="https://github.com/user-attachments/assets/3f0e087a-34c6-474f-8928-d6a546e955e1" />

### 🔹 Segment Analysis Charts
<img width="1904" height="669" alt="image" src="https://github.com/user-attachments/assets/aecb180a-52b2-462a-9bbf-0f3b36c7cc3c" />
<img width="1859" height="611" alt="image" src="https://github.com/user-attachments/assets/abc9d685-6a14-4eea-a8bd-115c388bb193" />

This interactive machine learning project allows businesses to **segment customers** and **predict whether they are likely to make a purchase next month**, using key behavioral metrics such as total orders, spending, and recency.

Built with:

- Python (Pandas, Scikit-learn, Seaborn, Joblib)
- Streamlit (interactive web app framework)

---

## 📊 Features

🔢 **Customer Input Panel**  
Users can enter customer metrics:

- Total Orders
- Total Spend
- Recency (in days)

🧠 **Prediction**  
The app predicts:

- 📦 **Customer Segment**: High Value / Low Value
- 🔮 **Next Month Purchase Prediction**: Yes / No

📈 **Segment Behavior Analysis (1-Click Visuals)**  
Clicking a button reveals line charts to analyze:

- Number of customers in each segment
- Average spend per segment
- Recency vs spend (by segment)
- Purchase probability per segment

📂 **Data Preview**  
Quick view of the customer dataset used to train the model.

---

## 🧠 ML Models Used

- **KMeans Clustering** – For customer segmentation
- **Logistic Regression** – For predicting next month’s purchase
- **StandardScaler** – For feature normalization

All models are pre-trained and stored in the `/models` directory:

- `scaler.pkl`
- `kmeans_model.pkl`
- `classifier_model.pkl`
- `customer_features.csv` (input data)

---

## 🖥️ How to Run the App

1. Clone this repo or download the files
2. Ensure Python 3.x is installed
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch the app:
   ```bash
    streamlit run streamlit_app.py
   ```

## ✨ Highlights

- Realistic pipeline using real-world data
- Combines **unsupervised + supervised learning**
- Fast, interactive **Streamlit UI**

---

## 📌 Author

**Sri Harshitha Adabala**

---

## 📎 Tags

`#MachineLearning` `#CustomerSegmentation` `#EcommerceAnalytics` `#Python` `#Streamlit` `#Clustering` `#Classification`
