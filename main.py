import pandas as pd
import numpy as np
import os
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

print("👀 Starting script...")

# Load Excel
file_path = "Copy of Inventory_Prediction_System(1).xlsx"
df = pd.read_excel(file_path)

# Clean Product_IDs (e.g., '2,001' -> '2001')
df['Product_ID'] = df['Product_ID'].astype(str).str.replace(",", "").str.strip()

# Simulate Quantity (only for testing, remove if real data exists)
df['Quantity'] = np.random.randint(1, 10, size=len(df))

# Date and Time Features
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Order Time'] = pd.to_datetime(df['Order Time'], format='%H:%M:%S', errors='coerce').dt.time
df['Hour'] = df['Order Time'].apply(lambda x: x.hour if pd.notnull(x) else np.nan)
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.day_name()
df['Is_Weekend'] = df['Weekday'].isin(['Saturday', 'Sunday']).astype(int)

# Stock Level & Discount Mapping
stock_map = {'In Stock': 2, 'Low Stock': 1}
df['Stock_Level'] = df['Stock'].map(stock_map).fillna(0)
df['Has_Discount'] = df['Discount Amt'].apply(lambda x: 1 if x > 0 else 0)

# Daily and Avg Sales
daily_sales = df.groupby(['Product_ID', 'Date'])['Quantity'].sum().reset_index()
daily_sales.rename(columns={'Quantity': 'Daily_Sales'}, inplace=True)
df = pd.merge(df, daily_sales, on=['Product_ID', 'Date'], how='left')

avg_sales = df.groupby('Product_ID')['Quantity'].mean().reset_index()
avg_sales.rename(columns={'Quantity': 'Avg_Sales_Per_Order'}, inplace=True)
df = pd.merge(df, avg_sales, on='Product_ID', how='left')

# Model Prep
features = ['Hour', 'Day', 'Month', 'Is_Weekend', 'Stock_Level', 'Has_Discount', 'Avg_Sales_Per_Order']
target = 'Daily_Sales'
df_model = df[features + [target]].dropna()
X = df_model[features]
y = df_model[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Models
print("\n✅ Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("📊 LR RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("📊 LR MAPE:", mean_absolute_percentage_error(y_test, y_pred))

print("\n🌲 Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("🌲 RF RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print("🌲 RF MAPE:", mean_absolute_percentage_error(y_test, rf_preds))

print("\n⚡ Training XGBoost...")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
print("⚡ XGB RMSE:", np.sqrt(mean_squared_error(y_test, xgb_preds)))
print("⚡ XGB MAPE:", mean_absolute_percentage_error(y_test, xgb_preds))

# Generate full prediction from model data
df['Predicted_Sales_XGB'] = np.nan
df.loc[df_model.index, 'Predicted_Sales_XGB'] = xgb.predict(X)

print("\n✅ ALL MODELS TRAINED & PREDICTIONS GENERATED")

# Export prediction summary
predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'Pred_LR': y_pred,
    'Pred_RF': rf_preds,
    'Pred_XGB': xgb_preds
})
predictions_df.to_excel("predicted_inventory_output.xlsx", index=False)
print("📁 Exported overall predictions to: predicted_inventory_output.xlsx")

# 🔍 Ask for Product ID
product_id_input = input("\n🔢 Enter Product ID to check stock prediction and reorder suggestion: ").strip()

if product_id_input not in df['Product_ID'].unique():
    print("❌ Product ID not found.")
    exit()

product_data = df[df['Product_ID'] == product_id_input].copy()

if product_data['Predicted_Sales_XGB'].isnull().all():
    print("⚠️ Product exists, but no predictions available (missing values or dropped during training).")
    exit()

# Reorder Logic
product_data['Reorder_Required'] = product_data['Predicted_Sales_XGB'] > product_data['Stock_Level']
product_data['Reorder_Quantity'] = (product_data['Predicted_Sales_XGB'] - product_data['Stock_Level']).apply(lambda x: max(0, round(x)))

# Export Product Report — Safe Versioning
base_filename = f"reorder_plan_product_{product_id_input}"
filename = f"{base_filename}.xlsx"
version = 1

while True:
    try:
        product_data.to_excel(filename, index=False)
        print(f"\n✅ Reorder plan exported to: {filename}")
        try:
            os.startfile(filename)
        except:
            print("📂 Could not auto-open the file, but it's saved.")
        break
    except PermissionError:
        version += 1
        filename = f"{base_filename}_v{version}.xlsx"
        time.sleep(0.5)
    