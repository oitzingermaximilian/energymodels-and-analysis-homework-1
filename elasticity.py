import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def read_hourly_prices(csv_file_path):
    """
    Reads a CSV file with one column ("AT") containing 8,760 hourly prices.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        pandas.Series: Hourly prices as a Series (or None if error occurs).
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Validate the data
        if "AT" not in df.columns:
            raise ValueError("CSV must contain a column named 'AT'")

        if len(df) != 8760:
            print(f"Warning: Expected 8,760 rows, got {len(df)} rows.")

        return df["AT"]  # Return the price column as a Series

    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def read_load_profiles(excel_file_path, sheet_name=0):
    """
    Reads an Excel file and extracts the 'Value_ScaleTo100' column (load profiles).

    Args:
        excel_file_path (str): Path to the Excel file.
        sheet_name (str/int): Sheet name or index (default: first sheet).

    Returns:
        pandas.Series: Load profile values as a Series (or None if error occurs).
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

        # Validate the column exists
        if "Value_ScaleTo100" not in df.columns:
            raise ValueError("Column 'Value_ScaleTo100' not found in the Excel sheet.")

        return df["Value_ScaleTo100"]  # Return the load profile column

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Example usage:
load = read_load_profiles("data_assignement_1/hourly_load_profile_electricity_AT_2023.xlsx")
prices = read_hourly_prices("data_assignement_1/preise2023.csv")
print(load)
print(prices)

# ----------------------------------------------------------------------------
# 2. Clean Data (Handle zeros/negatives for robust analysis)
# ----------------------------------------------------------------------------
# For non-log methods, we can keep negatives but add small offset to avoid division by zero
load_clean = load.copy()
price_clean = prices.copy()
load_clean = load_clean + 1e-6  # Small offset to avoid exact zeros
price_clean = (price_clean + 1e-6) * 10 #to convert to €/Mwh

# ----------------------------------------------------------------------------
# 1. Data Validation
# ----------------------------------------------------------------------------
print("Data Validation:")
print(f"- Price range: {price_clean.min():.2f} to {price_clean.max():.2f} €/MWh")
print(f"- Load range: {load_clean.min():.2f} to {load_clean.max():.2f} MWh")

# ----------------------------------------------------------------------------
# 2. Visual Inspection (Critical!)
# ----------------------------------------------------------------------------
plt.figure(figsize=(12,5))
plt.scatter(price_clean, load_clean, alpha=0.1, s=10)
plt.xlabel("Price (€/MWh)")
plt.ylabel("Load (MWh)")
plt.title("Raw Price-Load Relationship")
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------------
# 3. Filtered Arc Elasticity (Most Reliable)
# ----------------------------------------------------------------------------
def calculate_elasticity(prices, loads):
    """Safe elasticity calculation with outlier filtering"""
    elasticities = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0.1:  # Ignore near-zero prices
            dp = prices[i] - prices[i-1]
            dq = loads[i] - loads[i-1]
            if abs(dp/prices[i-1]) > 0.001:  # Ignore tiny changes
                elasticity = (dq/loads[i-1]) / (dp/prices[i-1])
                if -10 < elasticity < 10:  # Filter extreme values
                    elasticities.append(elasticity)
    return np.nanmedian(elasticities)

arc_alpha = calculate_elasticity(price_clean.values, load_clean.values)

# ----------------------------------------------------------------------------
# 4. Robust Regression (Alternative Check)
# ----------------------------------------------------------------------------
# Normalize data first to avoid scale issues
price_norm = (price_clean - price_clean.mean()) / price_clean.std()
load_norm = (load_clean - load_clean.mean()) / load_clean.std()

model = sm.RLM(load_norm, sm.add_constant(price_norm),
              M=sm.robust.norms.HuberT()).fit()
reg_alpha = model.params[1]  # Slope for normalized data

# Convert back to original scale
real_world_alpha = reg_alpha * (load_clean.std() / price_clean.std())

# ----------------------------------------------------------------------------
# 5. Results & Diagnostics
# ----------------------------------------------------------------------------
print(f"""
🔍 Final Elasticity Estimates:
---------------------------------
1. Filtered Arc Elasticity (Median):
   α = {arc_alpha:.4f}

2. Robust Regression (Real-World Units):
   α = {real_world_alpha:.4f}

Expected Range for Electricity Markets:
- Short-term: -0.1 to -0.5
- Long-term: -0.5 to -1.2
""")

# Residuals check
if hasattr(model, 'resid'):
    plt.figure(figsize=(12,4))
    plt.scatter(model.predict(), model.resid, alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals Plot")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.show()