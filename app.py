# ========================================
# 🚗 EV ADOPTION & FUEL PRICE TREND ANALYSIS (Enhanced Linear Regression)
# ========================================

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

# ========================================
# SETTINGS
# ========================================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("🚗 EV ADOPTION & FUEL PRICE TREND ANALYSIS (Enhanced Linear Model)")
print("="*70)

# ========================================
# 1. DATA LOADING
# ========================================
print("\n📂 Loading datasets...")
try:
    ev_data = pd.read_csv('ev_registrations.csv')
    fuel_data = pd.read_csv('fuel_prices.csv')
    print("✅ Datasets loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Please ensure 'ev_registrations.csv' and 'fuel_prices.csv' are available.")
    exit()

# ========================================
# 2. DATA PREPROCESSING
# ========================================
print("\n🔧 Preprocessing data...")

ev_data['Date'] = pd.to_datetime(ev_data['Date'])
fuel_data['Date'] = pd.to_datetime(fuel_data['Date'])

merged_data = pd.merge(ev_data, fuel_data, on='Date', how='inner')
merged_data['Year'] = merged_data['Date'].dt.year
merged_data['Month'] = merged_data['Date'].dt.month
merged_data['Days_Since_Start'] = (merged_data['Date'] - merged_data['Date'].min()).dt.days

# Feature engineering
merged_data['Avg_Fuel_Price'] = (merged_data['Petrol_Price'] + merged_data['Diesel_Price']) / 2
merged_data['Fuel_Price_Diff'] = merged_data['Petrol_Price'] - merged_data['Diesel_Price']

print(f"✅ Merged dataset shape: {merged_data.shape}")

# ========================================
# 3. CORRELATION ANALYSIS (Improved Heatmap)
# ========================================
print("\n📊 Correlation Analysis...")

corr = merged_data[['EV_Registrations', 'Petrol_Price', 'Diesel_Price',
                    'Days_Since_Start', 'Avg_Fuel_Price', 'Fuel_Price_Diff']].corr()
print(corr)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdYlGn', center=0,
            linewidths=1, linecolor='black', cbar=True,
            annot_kws={"size":12, "weight":"bold"})
plt.xticks(rotation=45, ha='right', fontsize=11, weight='bold')
plt.yticks(fontsize=11, weight='bold')
plt.title("🔍 Correlation Heatmap: EV Registrations vs Fuel Prices", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_heatmap_improved.png", dpi=300)
plt.show()


# ========================================
# 4. MODEL PREPARATION
# ========================================
print("\n⚙️ Preparing data for regression...")

X = merged_data[['Petrol_Price', 'Diesel_Price', 'Avg_Fuel_Price', 'Fuel_Price_Diff', 'Days_Since_Start']]
y = np.log1p(merged_data['EV_Registrations'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========================================
# 5. MODEL TRAINING
# ========================================
print("\n🤖 Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ========================================
# 6. MODEL EVALUATION
# ========================================
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

print("\n" + "="*70)
print("📈 MODEL PERFORMANCE (Linear Regression)")
print("="*70)
print(f"Mean Squared Error (MSE):       {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"Mean Absolute Error (MAE):      {mae:,.2f}")
print(f"R² Score:                       {r2:.4f}")
print(f"Cross-Validation Mean R²:       {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ========================================
# 7. VISUALIZATION - Actual vs Predicted
# ========================================
plt.figure(figsize=(12, 6))
plt.scatter(y_true, y_pred, alpha=0.7, color='#2980b9', edgecolors='black')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Fit')
plt.xlabel('Actual EV Registrations', fontsize=12, fontweight='bold')
plt.ylabel('Predicted EV Registrations', fontsize=12, fontweight='bold')
plt.title('🎯 Actual vs Predicted EV Registrations (Linear Model)', fontsize=15, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted_linear.png', dpi=300)
print("✅ Saved: actual_vs_predicted_linear.png")
plt.show()

# ========================================
# 8. TREND VISUALIZATION
# ========================================
plt.figure(figsize=(12,6))
plt.plot(merged_data['Date'], merged_data['EV_Registrations'], label='EV Registrations', color='#27ae60', linewidth=2)
plt.plot(merged_data['Date'], merged_data['Avg_Fuel_Price']*100, label='Avg Fuel Price (x100)', color='#e67e22', linewidth=2)
plt.title("📅 EV Growth vs Fuel Price Trend", fontsize=15, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Value / Scaled (x100 for Fuel Price)")
plt.legend()
plt.tight_layout()
plt.savefig('trend_analysis.png', dpi=300)
print("✅ Saved: trend_analysis.png")
plt.show()

# ========================================
# 9. FUTURE PREDICTIONS
# ========================================
print("\n🔮 Predicting future EV adoption...")
max_days = merged_data['Days_Since_Start'].max()

future_data = pd.DataFrame({
    'Petrol_Price': [100, 105, 110, 115, 120],
    'Diesel_Price': [90, 95, 100, 105, 110],
})
future_data['Avg_Fuel_Price'] = (future_data['Petrol_Price'] + future_data['Diesel_Price']) / 2
future_data['Fuel_Price_Diff'] = future_data['Petrol_Price'] - future_data['Diesel_Price']
future_data['Days_Since_Start'] = [max_days + 30*i for i in range(1, 6)]

future_scaled = scaler.transform(future_data[X.columns])
future_pred_log = model.predict(future_scaled)
future_data['Predicted_EV_Registrations'] = np.expm1(future_pred_log).round(0)
future_data['Predicted_EV_Registrations'] = future_data['Predicted_EV_Registrations'].clip(lower=0)

print("\n📈 Future EV Adoption Predictions:")
print(future_data[['Petrol_Price', 'Diesel_Price', 'Predicted_EV_Registrations']])

plt.figure(figsize=(12,6))
plt.plot(future_data['Petrol_Price'], future_data['Predicted_EV_Registrations'], marker='o', linewidth=3, color='#16a085')
plt.title("🔮 Future EV Adoption Predictions vs Fuel Price", fontsize=15, fontweight='bold')
plt.xlabel("Petrol Price (₹/Litre)")
plt.ylabel("Predicted EV Registrations")
plt.grid(True, alpha=0.3)
for x, y in zip(future_data['Petrol_Price'], future_data['Predicted_EV_Registrations']):
    plt.text(x, y, f'{int(y):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('future_predictions_linear.png', dpi=300)
print("✅ Saved: future_predictions_linear.png")
plt.show()

# ========================================
# 10. INSIGHTS
# ========================================
print("\n" + "="*70)
print("💡 INSIGHTS & SUMMARY")
print("="*70)

corr_value = corr.loc['EV_Registrations', 'Avg_Fuel_Price']
growth_rate = (merged_data['EV_Registrations'].iloc[-1] - merged_data['EV_Registrations'].iloc[0]) / merged_data['EV_Registrations'].iloc[0] * 100

print(f"\n✅ R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
print(f"✅ Mean Absolute Error: ±{mae:,.0f} registrations")
print(f"✅ EV growth from {merged_data['Date'].min().year}–{merged_data['Date'].max().year}: {growth_rate:.2f}%")
print(f"✅ Correlation between EV adoption & Avg Fuel Price: {corr_value:.2f}")

print("\n🎯 As petrol price increases from ₹100 → ₹120:")
print(f"   EV registrations are predicted to rise from "
      f"{int(future_data['Predicted_EV_Registrations'].iloc[0]):,} "
      f"to {int(future_data['Predicted_EV_Registrations'].iloc[-1]):,}, "
      f"showing a positive association with fuel cost inflation.")

print("\n✅ Analysis Complete. Generated Files:")
print("   • correlation_heatmap.png")
print("   • actual_vs_predicted_linear.png")
print("   • trend_analysis.png")
print("   • future_predictions_linear.png")
print("="*70)

# ========================================
# 11. INFOGRAPHIC SUMMARY 
# ========================================

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

print("\n🖼️ Generating improved infographic...")

fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#f7f7f7')
gs = GridSpec(3, 2, height_ratios=[0.8, 4, 1.2], width_ratios=[1, 1], figure=fig)

ax_title = fig.add_subplot(gs[0, :])
ax_title.axis('off')
ax_title.text(
    0.5, 0.5,
    "EV Adoption & Fuel Price Insights Dashboard",
    fontsize=18, fontweight='bold', ha='center', va='center'
)

ax_metrics = fig.add_subplot(gs[1, 0])
ax_metrics.axis('off')
metrics_text = (
    f"• R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)\n"
    f"• Mean Absolute Error: ±{mae:,.0f}\n"
    f"• EV Growth: {growth_rate:.2f}% "
    f"({merged_data['Date'].min().year}–{merged_data['Date'].max().year})\n"
    f"• Correlation (EV vs Avg Fuel Price): {corr_value:.2f}"
)
ax_metrics.text(
    0.02, 0.95,
    metrics_text,
    fontsize=12, va='top',
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#e8f4fa", edgecolor="#2980b9")
)

ax_plot = fig.add_subplot(gs[1, 1])
ax_plot.plot(
    future_data['Petrol_Price'],
    future_data['Predicted_EV_Registrations'],
    marker='o', linewidth=2
)
ax_plot.set_title("Predicted EV Adoption vs Petrol Price", fontsize=12, fontweight='bold')
ax_plot.set_xlabel("Petrol Price (₹/Litre)")
ax_plot.set_ylabel("Predicted EV Registrations")
ax_plot.grid(alpha=0.3)

for x, y in zip(future_data['Petrol_Price'], future_data['Predicted_EV_Registrations']):
    ax_plot.text(x, y, f"{int(y):,}", fontsize=10, ha='center', va='bottom')

ax_footer = fig.add_subplot(gs[2, :])
ax_footer.axis('off')
footer_text = (
    f"As petrol price rises from ₹100 to ₹120, EV registrations are predicted "
    f"to increase from {int(future_data['Predicted_EV_Registrations'].iloc[0]):,} "
    f"to {int(future_data['Predicted_EV_Registrations'].iloc[-1]):,}."
)
ax_footer.text(
    0.5, 0.5,
    footer_text,
    fontsize=12, ha='center',
    bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff4e6")
)

infographic_path = "ev_fuel_infographic.png"
plt.tight_layout()
plt.savefig(infographic_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Infographic generated and saved as: {infographic_path}")


