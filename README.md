# 🛒 Customer Segmentation & Purchase Prediction (ML-based Streamlit App)

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
