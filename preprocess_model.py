import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Load CSVs
orders = pd.read_csv("data/Orders.csv")
details = pd.read_csv("data/Order Details.csv")
sales = pd.read_csv("data/Sales.csv")

# 🧩 Merge: Order Details + Orders (on Order ID)
merged = pd.merge(details, orders, on="Order ID", how="left")

# 🧩 Merge: Merged with Sales (on Category -> assumes category mapping to monthly target)
merged = pd.merge(merged, sales[['Category', 'Target']], on="Category", how="left")

# ✅ Check required columns
if 'Order Date' not in merged.columns or 'CustomerName' not in merged.columns or 'Amount' not in merged.columns:
    raise Exception("Missing required columns like 'Order Date', 'CustomerName', or 'Amount'")

# 🕒 Convert Order Date to datetime
merged['Order Date'] = pd.to_datetime(merged['Order Date'], errors='coerce')
merged.dropna(subset=['Order Date'], inplace=True)

# 📌 Create RFM features
latest_date = merged['Order Date'].max()

rfm = merged.groupby('CustomerName').agg({
    'Order ID': 'nunique',                  # Frequency
    'Amount': 'sum',                        # Monetary
    'Order Date': lambda x: (latest_date - x.max()).days  # Recency
}).reset_index()

rfm.columns = ['Customer Name', 'Total Orders', 'Total Spend', 'Recency']

# 🔍 Scaling before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm[['Total Orders', 'Total Spend', 'Recency']])

# 🎯 KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
rfm['Segment'] = kmeans.fit_predict(X_scaled)

# ✅ Create Target: Purchase Likely if Recency <= 30 days
rfm['Will Purchase Next Month'] = (rfm['Recency'] <= 30).astype(int)

# 🤖 Classification: Predict purchase likelihood
Xc = rfm[['Total Orders', 'Total Spend', 'Recency']]
yc = rfm['Will Purchase Next Month']
X_train, X_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 💾 Save outputs
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(clf, "models/classifier_model.pkl")
rfm.to_csv("models/customer_features.csv", index=False)

print("✅ Training completed!")
print("🎯 Classifier accuracy:", clf.score(X_test, y_test))
print("📁 Outputs saved in /models/")
