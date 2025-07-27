# # streamlit_app.py
# import streamlit as st
# import pandas as pd
# import joblib

# st.title("🛒 Customer Segmentation & Purchase Prediction (ML-based)")

# # Input
# st.sidebar.header("Enter Customer Metrics")
# total_orders = st.sidebar.number_input("Total Orders", min_value=0)
# total_spend = st.sidebar.number_input("Total Spend", min_value=0.0)
# recency = st.sidebar.number_input("Recency (days since last purchase)", min_value=0)

# if st.sidebar.button("Predict"):
#     scaler = joblib.load("models/scaler.pkl")
#     kmeans = joblib.load("models/kmeans_model.pkl")
#     clf = joblib.load("models/classifier_model.pkl")
    
#     df = pd.DataFrame([[total_orders, total_spend, recency]],
#                       columns=['Total Orders', 'Total Spend', 'Recency'])
#     scaled = scaler.transform(df)
#     seg = kmeans.predict(scaled)[0]
#     will_buy = clf.predict(df)[0]

#     segment_label = "High Value" if seg == 1 else "Low Value"
#     purchase_label = "Yes" if will_buy == 1 else "No"

#     st.success(f"🎯 Segment: {segment_label}")
#     st.success(f"📦 Likely to Purchase Next Month: {purchase_label}")

#     if segment_label == "High Value":
#         st.markdown("""
#         ### 💎 High Value Customer
#         - Frequently shops and spends more on the platform.
#         - Recent engagement.
#         - Suggest targeted loyalty programs or VIP offers.
#         """)
#     else:
#         st.markdown("""
#         ### 🧊 Low Value Customer
#         - Low engagement or infrequent purchasing history.
#         - Consider re‑engagement offers, surveys or discounts.
#         """)

#     if purchase_label == "Yes":
#         st.info("✅ This customer is likely to purchase next month.")
#     else:
#         st.warning("⚠️ May not purchase next month — recommend outreach.")

# st.subheader("📂 Preview customer feature sample")
# dataframe = pd.read_csv("models/customer_features.csv")
# st.write(dataframe.head(5))


# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🛒 Customer Segmentation & Purchase Prediction (ML-based)")

# Sidebar Input
st.sidebar.header("Enter Customer Metrics")
total_orders = st.sidebar.number_input("Total Orders", min_value=0)
total_spend = st.sidebar.number_input("Total Spend", min_value=0.0)
recency = st.sidebar.number_input("Recency (days since last purchase)", min_value=0)

if st.sidebar.button("Predict"):
    scaler = joblib.load("models/scaler.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    clf = joblib.load("models/classifier_model.pkl")

    df = pd.DataFrame([[total_orders, total_spend, recency]],
                      columns=['Total Orders', 'Total Spend', 'Recency'])
    scaled = scaler.transform(df)
    seg = kmeans.predict(scaled)[0]
    will_buy = clf.predict(df)[0]

    segment_label = "High Value" if seg == 1 else "Low Value"
    purchase_label = "Yes" if will_buy == 1 else "No"

    st.success(f"🎯 Segment: {segment_label}")
    st.success(f"📦 Likely to Purchase Next Month: {purchase_label}")

    if segment_label == "High Value":
        st.markdown("""
        ### 💎 High Value Customer
        - Frequently shops and spends more on the platform.
        - Recent engagement.
        - Suggest targeted loyalty programs or VIP offers.
        """)
    else:
        st.markdown("""
        ### 🧊 Low Value Customer
        - Low engagement or infrequent purchasing history.
        - Consider re‑engagement offers, surveys or discounts.
        """)

    if purchase_label == "Yes":
        st.info("✅ This customer is likely to purchase next month.")
    else:
        st.warning("⚠️ May not purchase next month — recommend outreach.")

# Preview data
st.subheader("📂 Preview customer feature sample")
dataframe = pd.read_csv("models/customer_features.csv")
st.write(dataframe.head(5))

# Segment Analysis Button
if st.button("📊 Analyze Customer Segments"):
    st.subheader("📈 Segment Analysis Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        seg_count = dataframe['Segment'].value_counts().sort_index()
        sns.lineplot(x=seg_count.index, y=seg_count.values, marker='o', ax=ax1)
        ax1.set_title("Number of Customers per Segment")
        ax1.set_xlabel("Segment")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        avg_spend = dataframe.groupby('Segment')['Total Spend'].mean().reset_index()
        sns.lineplot(x='Segment', y='Total Spend', data=avg_spend, marker='o', ax=ax2)
        ax2.set_title("Average Spend by Segment")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(4, 3))
        sns.lineplot(x=dataframe['Recency'], y=dataframe['Total Spend'],
                     hue=dataframe['Segment'], ax=ax3, marker='o')
        ax3.set_title("Recency vs Spend by Segment")
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(4, 3))
        purchase_prob = dataframe.groupby('Segment')['Will Purchase Next Month'].mean().reset_index()
        sns.lineplot(x='Segment', y='Will Purchase Next Month', data=purchase_prob, marker='o', ax=ax4)
        ax4.set_title("Purchase Probability by Segment")
        st.pyplot(fig4)
