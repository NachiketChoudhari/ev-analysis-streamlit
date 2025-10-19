# ========================================
# 🚗 Streamlit EV Analytics Dashboard (Complete)
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ========================================
# Page config
# ========================================
st.set_page_config(page_title="EV Analytics Dashboard", layout="wide")
sns.set_style("whitegrid")

# ========================================
# Load and preprocess data
# ========================================
ev_data = pd.read_csv('ev_registrations.csv')
fuel_data = pd.read_csv('fuel_prices.csv')

ev_data['Date'] = pd.to_datetime(ev_data['Date'])
fuel_data['Date'] = pd.to_datetime(fuel_data['Date'])

merged_data = pd.merge(ev_data, fuel_data, on='Date')
merged_data['Avg_Fuel_Price'] = (merged_data['Petrol_Price'] + merged_data['Diesel_Price']) / 2
merged_data['Fuel_Price_Diff'] = merged_data['Petrol_Price'] - merged_data['Diesel_Price']
merged_data['Days_Since_Start'] = (merged_data['Date'] - merged_data['Date'].min()).dt.days

# ========================================
# Train Linear Regression Model
# ========================================
X = merged_data[['Petrol_Price','Diesel_Price','Avg_Fuel_Price','Fuel_Price_Diff','Days_Since_Start']]
y = np.log1p(merged_data['EV_Registrations'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression().fit(X_scaled, y)
y_pred = np.expm1(model.predict(X_scaled))
r2 = model.score(X_scaled, y)
mae = mean_absolute_error(np.expm1(y), y_pred)

# ========================================
# Sidebar Navigation
# ========================================
dashboard = st.sidebar.selectbox(
    "Choose Dashboard", 
    [
        "Overview", 
        "Actual vs Predicted", 
        "Interactive Prediction", 
        "Future Trends", 
        "Insights & Infographics", 
        "Historical Data Explorer"
    ]
)

# ========================================
# 1️⃣ Overview Dashboard (Expanded)
# ========================================
if dashboard == "Overview":
    st.title("📊 EV Adoption Overview")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("Mean Absolute Error", f"{mae:.0f}")
    growth_rate = ((merged_data['EV_Registrations'].iloc[-1] - merged_data['EV_Registrations'].iloc[0])
                   / merged_data['EV_Registrations'].iloc[0] * 100)
    col3.metric("EV Growth (%)", f"{growth_rate:.2f}")
    
    # Descriptive text
    st.markdown("""
    **Overview:**  
    The EV adoption in the selected region shows a steady growth over the analyzed period.  
    Observing the trend with average fuel prices indicates a correlation between fuel costs and EV registrations.  
    High fuel prices appear to coincide with accelerated EV adoption, highlighting consumer sensitivity to petrol and diesel prices.  
    This overview provides insights into historical trends and serves as a foundation for forecasting future EV growth.
    """)
    
    # EV vs Fuel Price Plot
    st.subheader("EV Registrations vs Avg Fuel Price")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(merged_data['Date'], merged_data['EV_Registrations'], label='EV Registrations', color='green', lw=2)
    ax.plot(merged_data['Date'], merged_data['Avg_Fuel_Price']*100, label='Avg Fuel Price (x100)', color='orange', lw=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value / Scaled (x100)")
    ax.legend()
    st.pyplot(fig)
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr = merged_data[['EV_Registrations', 'Petrol_Price', 'Diesel_Price', 'Avg_Fuel_Price', 'Fuel_Price_Diff']].corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdYlGn', center=0, ax=ax)
    st.pyplot(fig)

# ========================================
# 2️⃣ Actual vs Predicted Dashboard
# ========================================
elif dashboard == "Actual vs Predicted":
    st.title("🎯 Actual vs Predicted EV Registrations")
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(np.expm1(y), y_pred, alpha=0.7, color='#2980b9')
    ax.plot([np.expm1(y).min(), np.expm1(y).max()], [np.expm1(y).min(), np.expm1(y).max()], 'r--', lw=2)
    ax.set_xlabel("Actual EV Registrations")
    ax.set_ylabel("Predicted EV Registrations")
    st.pyplot(fig)
    
    # Residuals
    st.subheader("Residuals Plot")
    residuals = np.expm1(y) - y_pred
    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(np.expm1(y), residuals, color='#e74c3c', alpha=0.7)
    ax.axhline(0, color='black', lw=2, linestyle='--')
    ax.set_xlabel("Actual EV Registrations")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

# ========================================
# 3️⃣ Interactive Prediction Dashboard
# ========================================
elif dashboard == "Interactive Prediction":
    st.title("🔮 Interactive EV Prediction")
    petrol_price = st.slider("Petrol Price (₹/Litre)", 80, 150, 110)
    diesel_price = st.slider("Diesel Price (₹/Litre)", 70, 140, 100)
    months_ahead = st.slider("Months Ahead for Prediction", 1, 12, 6)
    
    future_days = [merged_data['Days_Since_Start'].max() + 30*i for i in range(1, months_ahead+1)]
    future_df = pd.DataFrame({'Petrol_Price':[petrol_price]*months_ahead, 'Diesel_Price':[diesel_price]*months_ahead})
    future_df['Avg_Fuel_Price'] = (future_df['Petrol_Price'] + future_df['Diesel_Price'])/2
    future_df['Fuel_Price_Diff'] = future_df['Petrol_Price'] - future_df['Diesel_Price']
    future_df['Days_Since_Start'] = future_days
    
    future_scaled = scaler.transform(future_df[X.columns])
    future_df['Predicted_EV'] = np.expm1(model.predict(future_scaled)).round(0)
    
    st.subheader("Predicted EV Registrations Table")
    st.dataframe(future_df)
    
    st.subheader("Predicted EV Registrations Plot")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(future_df['Days_Since_Start'], future_df['Predicted_EV'], marker='o', color='#16a085', lw=2)
    ax.set_xlabel("Days Since Start")
    ax.set_ylabel("Predicted EV Registrations")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ========================================
# 4️⃣ Future Trends Dashboard
# ========================================
elif dashboard == "Future Trends":
    st.title("📈 Future EV Trend Simulation")
    st.write("Compare multiple petrol price scenarios to predict EV adoption.")
    
    months_ahead = st.slider("Months to Forecast", 1, 12, 6, key='ft_months')
    base_diesel_price = st.slider("Base Diesel Price (₹)", 70, 140, 100, key='ft_diesel')
    
    petrol_scenarios = {
        'Low Petrol Price': np.linspace(100, 105, months_ahead),
        'Medium Petrol Price': np.linspace(110, 120, months_ahead),
        'High Petrol Price': np.linspace(125, 140, months_ahead)
    }
    
    future_days = [merged_data['Days_Since_Start'].max() + 30*i for i in range(1, months_ahead+1)]
    forecast_df = pd.DataFrame({'Days_Since_Start': future_days})
    
    fig, ax = plt.subplots(figsize=(12,6))
    for scenario, petrol_prices in petrol_scenarios.items():
        temp_df = pd.DataFrame({
            'Petrol_Price': petrol_prices,
            'Diesel_Price': [base_diesel_price]*months_ahead
        })
        temp_df['Avg_Fuel_Price'] = (temp_df['Petrol_Price'] + temp_df['Diesel_Price']) / 2
        temp_df['Fuel_Price_Diff'] = temp_df['Petrol_Price'] - temp_df['Diesel_Price']
        temp_df['Days_Since_Start'] = future_days
        temp_scaled = scaler.transform(temp_df[X.columns])
        temp_df['Predicted_EV'] = np.expm1(model.predict(temp_scaled)).round(0)
        
        forecast_df[scenario] = temp_df['Predicted_EV']
        ax.plot(future_days, temp_df['Predicted_EV'], marker='o', lw=2, label=scenario)
    
    ax.set_xlabel("Days Since Start")
    ax.set_ylabel("Predicted EV Registrations")
    ax.set_title(f"EV Adoption Forecast for Next {months_ahead} Months")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Scenario Prediction Table")
    st.dataframe(forecast_df)

# ========================================
# 5️⃣ Insights & Infographics (Text-Based)
# ========================================
elif dashboard == "Insights & Infographics":
    st.title("💡 Insights & Infographics")

    # Safe growth_rate
    start_ev = merged_data['EV_Registrations'].iloc[0]
    end_ev = merged_data['EV_Registrations'].iloc[-1]
    growth_rate = ((end_ev - start_ev) / start_ev * 100) if start_ev != 0 else 0

    st.subheader("Summary Metrics")
    st.write(f"• R² Score: {r2:.4f}")
    st.write(f"• Mean Absolute Error: ±{mae:.0f}")
    st.write(f"• EV Growth: {growth_rate:.2f}%")
    
    # Correlation
    corr_value = merged_data[['EV_Registrations','Avg_Fuel_Price']].corr().iloc[0,1]
    st.subheader("Correlation with Avg Fuel Price")
    st.write(f"Correlation (EV vs Avg Fuel Price): {corr_value:.2f}")
    
    # Extended Text Insights
    st.subheader("Key Insights")
    st.markdown(f"""
    1. **Overall EV Growth:** From {start_ev:,} to {end_ev:,} registrations, total growth of {growth_rate:.2f}%.  
    2. **Fuel Price Sensitivity:** Correlation with Avg Fuel Price is {corr_value:.2f}, indicating moderate sensitivity.  
    3. **Trends Over Time:** EV adoption shows seasonal fluctuations and a general upward trend.  
    4. **Market Implications:** Rising fuel prices can be a strong driver for accelerated EV adoption.  
    5. **Forecasting Potential:** Historical patterns provide a reliable foundation for predictive modeling of future EV registrations.
    """)

# ========================================
# 6️⃣ Historical Data Explorer (with Insights)
# ========================================
elif dashboard == "Historical Data Explorer":
    st.title("📅 Historical EV Registrations vs Fuel Prices (2018–2024)")

    merged_data['Year'] = merged_data['Date'].dt.year
    selected_years = st.multiselect(
        "Select Years to Explore", 
        options=sorted(merged_data['Year'].unique()), 
        default=[2018,2019,2020,2021,2022,2023,2024]
    )

    if selected_years:
        filtered_data = merged_data[merged_data['Year'].isin(selected_years)]

        st.subheader("EV Registrations vs Fuel Prices Table")
        st.dataframe(filtered_data[['Date', 'EV_Registrations', 'Petrol_Price', 'Diesel_Price', 'Avg_Fuel_Price']].reset_index(drop=True))

        st.subheader("EV Registrations vs Fuel Prices Plot")
        fig, ax = plt.subplots(figsize=(12,6))
        for year in selected_years:
            year_data = filtered_data[filtered_data['Year'] == year]
            ax.plot(year_data['Date'], year_data['EV_Registrations'], marker='o', label=f"EV {year}")
            ax.plot(year_data['Date'], year_data['Avg_Fuel_Price']*100, marker='x', linestyle='--', label=f"Fuel Price {year}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value / Scaled (x100)")
        ax.set_title("EV Registrations vs Average Fuel Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Insights
        st.subheader("Key Insights")
        total_ev = filtered_data['EV_Registrations'].sum()
        avg_fuel_price = filtered_data['Avg_Fuel_Price'].mean()
        start_ev = filtered_data['EV_Registrations'].iloc[0]
        end_ev = filtered_data['EV_Registrations'].iloc[-1]
        growth_rate = ((end_ev - start_ev)/start_ev * 100) if start_ev != 0 else 0
        corr_value = filtered_data[['EV_Registrations','Avg_Fuel_Price']].corr().iloc[0,1] if 'Avg_Fuel_Price' in filtered_data.columns else 0

        st.markdown(f"""
        1. **Total EV Registrations (Selected Years):** {total_ev:,}  
        2. **EV Growth Rate:** {growth_rate:.2f}% from {start_ev:,} to {end_ev:,} registrations  
        3. **Average Fuel Price:** ₹{avg_fuel_price:.2f}  
        4. **Correlation (EV vs Avg Fuel Price):** {corr_value:.2f}  
        5. **Trend Observation:** EV registrations generally increase over time, with higher adoption observed during periods of elevated fuel prices.
        """)
    else:
        st.warning("Please select at least one year to display data.")
