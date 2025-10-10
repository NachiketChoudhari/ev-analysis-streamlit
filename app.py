import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Streamlit Config
st.set_page_config(page_title="EV Adoption vs Fuel Prices (EDA & Regression)",
                   layout="wide",
                   page_icon="🚗")

sns.set_style("whitegrid")

# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.title("🚗 EV Adoption & Fuel Price Trend Analysis")
st.markdown("### 🔍 Exploratory Data Analysis & Linear Regression Model using Python, Pandas, Seaborn, NumPy, and Scikit-learn")

# ---------------------------------------------
# DATA LOADING
# ---------------------------------------------
st.header("📂 Step 1: Load Datasets")

try:
    ev_data = pd.read_csv('ev_registrations.csv')
    fuel_data = pd.read_csv('fuel_prices.csv')
    st.success("✅ Datasets loaded successfully!")
except FileNotFoundError:
    st.error("❌ Missing files! Please ensure 'ev_registrations.csv' and 'fuel_prices.csv' are present.")
    st.stop()

# Display Data
col1, col2 = st.columns(2)
with col1:
    st.subheader("EV Registrations Data")
    st.dataframe(ev_data.head())
with col2:
    st.subheader("Fuel Prices Data")
    st.dataframe(fuel_data.head())

# ---------------------------------------------
# DATA PREPROCESSING
# ---------------------------------------------
st.header("🔧 Step 2: Data Preprocessing")

ev_data['Date'] = pd.to_datetime(ev_data['Date'])
fuel_data['Date'] = pd.to_datetime(fuel_data['Date'])
merged_data = pd.merge(ev_data, fuel_data, on='Date', how='inner')

# Feature Engineering
merged_data['Year'] = merged_data['Date'].dt.year
merged_data['Month'] = merged_data['Date'].dt.month
merged_data['Days_Since_Start'] = (merged_data['Date'] - merged_data['Date'].min()).dt.days
merged_data['Avg_Fuel_Price'] = (merged_data['Petrol_Price'] + merged_data['Diesel_Price']) / 2
merged_data['Fuel_Price_Diff'] = merged_data['Petrol_Price'] - merged_data['Diesel_Price']
merged_data['Petrol_Price_Sq'] = merged_data['Petrol_Price'] ** 2
merged_data['Time_Fuel_Interaction'] = merged_data['Days_Since_Start'] * merged_data['Avg_Fuel_Price']

st.write("✅ Merged dataset with engineered features:")
st.dataframe(merged_data.head())

# ---------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ---------------------------------------------
st.header("📊 Step 3: Exploratory Data Analysis")

# Correlation
corr_cols = ['EV_Registrations', 'Petrol_Price', 'Diesel_Price', 
             'Days_Since_Start', 'Avg_Fuel_Price', 'Fuel_Price_Diff']
corr_matrix = merged_data[corr_cols].corr()

st.subheader("🔗 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=1, ax=ax)
st.pyplot(fig)

# Time trends
st.subheader("📈 EV Registrations Over Time")
fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=merged_data, x='Date', y='EV_Registrations', ax=ax, color='#2ecc71')
ax.set_title('EV Registrations Over Time')
st.pyplot(fig)

st.subheader("⛽ Fuel Prices Over Time")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(merged_data['Date'], merged_data['Petrol_Price'], label='Petrol', color='#e74c3c')
ax.plot(merged_data['Date'], merged_data['Diesel_Price'], label='Diesel', color='#3498db')
ax.legend()
st.pyplot(fig)

# Scatter comparison
st.subheader("🚗 EV Registrations vs Fuel Prices")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.scatterplot(x='Petrol_Price', y='EV_Registrations', data=merged_data, color='#e67e22', ax=ax)
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(x='Diesel_Price', y='EV_Registrations', data=merged_data, color='#16a085', ax=ax)
    st.pyplot(fig)

# ---------------------------------------------
# LINEAR REGRESSION MODELS
# ---------------------------------------------
st.header("🤖 Step 4: Linear Regression Modeling")

# Feature sets
X_basic = merged_data[['Petrol_Price', 'Diesel_Price', 'Days_Since_Start']]
y = merged_data['EV_Registrations']

X_train, X_test, y_train, y_test = train_test_split(X_basic, y, test_size=0.2, random_state=42)
model_basic = LinearRegression().fit(X_train, y_train)
y_pred = model_basic.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📊 Basic Model Performance")
st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**Mean Absolute Error:** {mae:,.0f}")
st.write(f"**Root Mean Squared Error:** {np.sqrt(mse):,.0f}")

# Actual vs Predicted Plot
st.subheader("🎯 Actual vs Predicted EV Registrations")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color="#9b59b6", alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel("Actual EV Registrations")
ax.set_ylabel("Predicted EV Registrations")
st.pyplot(fig)

# ---------------------------------------------
# ENHANCED MODEL
# ---------------------------------------------
st.header("🚀 Step 5: Enhanced Linear Regression Model")

X_enh = merged_data[['Petrol_Price', 'Diesel_Price', 'Days_Since_Start',
                     'Avg_Fuel_Price', 'Fuel_Price_Diff', 
                     'Petrol_Price_Sq', 'Time_Fuel_Interaction']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_enh)

X_train_enh, X_test_enh, y_train_enh, y_test_enh = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model_enh = LinearRegression().fit(X_train_enh, y_train_enh)
y_pred_enh = model_enh.predict(X_test_enh)

mse_enh = mean_squared_error(y_test_enh, y_pred_enh)
r2_enh = r2_score(y_test_enh, y_pred_enh)
mae_enh = mean_absolute_error(y_test_enh, y_pred_enh)

st.write(f"**Enhanced Model R²:** {r2_enh:.4f}")
st.write(f"**MAE:** {mae_enh:,.0f}")
st.write(f"**RMSE:** {np.sqrt(mse_enh):,.0f}")

# Model Comparison
st.subheader("⚖️ Model Comparison")
comparison = pd.DataFrame({
    "Metric": ["R² Score", "MAE", "RMSE"],
    "Basic Model": [r2, mae, np.sqrt(mse)],
    "Enhanced Model": [r2_enh, mae_enh, np.sqrt(mse_enh)]
})
st.dataframe(comparison)

# ---------------------------------------------
# FUTURE PREDICTIONS
# ---------------------------------------------
st.header("🔮 Step 6: Future EV Adoption Prediction")

max_days = merged_data['Days_Since_Start'].max()
future_data = pd.DataFrame({
    'Petrol_Price': [110, 115, 120, 125, 130],
    'Diesel_Price': [95, 100, 105, 110, 115],
    'Days_Since_Start': [max_days + 30*i for i in range(1, 6)]
})
future_data['Avg_Fuel_Price'] = (future_data['Petrol_Price'] + future_data['Diesel_Price']) / 2
future_data['Fuel_Price_Diff'] = future_data['Petrol_Price'] - future_data['Diesel_Price']
future_data['Petrol_Price_Sq'] = future_data['Petrol_Price'] ** 2
future_data['Time_Fuel_Interaction'] = future_data['Days_Since_Start'] * future_data['Avg_Fuel_Price']

future_scaled = scaler.transform(future_data[X_enh.columns])
future_data['Predicted_EV_Registrations'] = model_enh.predict(future_scaled)

st.dataframe(future_data[['Petrol_Price', 'Diesel_Price', 'Predicted_EV_Registrations']])

# Plot predictions
fig, ax = plt.subplots()
ax.plot(future_data['Petrol_Price'], future_data['Predicted_EV_Registrations'],
        marker='o', linewidth=3, color='#27ae60')
ax.set_title("Predicted EV Registrations vs Fuel Price Scenarios")
ax.set_xlabel("Petrol Price (₹/L)")
ax.set_ylabel("Predicted EV Registrations")
st.pyplot(fig)

# ---------------------------------------------
# INSIGHTS
# ---------------------------------------------
st.header("💡 Step 7: Key Insights")
st.markdown(f"""
- **EV Adoption shows strong positive correlation with Time** (r = {corr_matrix.loc['EV_Registrations','Days_Since_Start']:.2f})
- **Fuel Prices are moderately correlated** with EV registrations.
- The **Enhanced Model** performs better (R² = {r2_enh:.3f}) than the Basic Model (R² = {r2:.3f})
- As petrol prices rise from ₹110 to ₹130, **EV registrations increase by approximately {((future_data['Predicted_EV_Registrations'].iloc[-1]/future_data['Predicted_EV_Registrations'].iloc[0]-1)*100):.1f}%**.
""")

st.success("✅ Analysis complete! Explore interactive charts and predictions above.")
