import pandas as pd
import numpy as np
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
load_profile = read_load_profiles("data_assignement_1/hourly_load_profile_electricity_AT_2023.xlsx")
prices = read_hourly_prices("data_assignement_1/preise2023.csv")
print(load_profile)
print(prices)

# 2. Validate Data
assert len(prices) == 8760, "Price data must have 8,760 hourly values"
assert len(load) == 8760, "Load data must have 8,760 hourly values"

# 3. Log-Transform for Elasticity Model
df = pd.DataFrame({
    "ln_price": np.log(prices),
    "ln_load": np.log(load)
}).dropna()  # Remove missing values

# 4. Run Regression (Elasticity Model)
X = sm.add_constant(df["ln_price"])  # Adds intercept (ln(C))
model = sm.OLS(df["ln_load"], X).fit()

# 5. Print Key Results
print(model.summary())  # Full regression report
print("\n🔥 Key Results:")
print(f"Price Elasticity (𝛼): {model.params['ln_price']:.4f}")
print(f"P-value: {model.pvalues['ln_price']:.4f}")
print(f"Demand at $1/MWh (C): {np.exp(model.params['const']):.2f}")

# 6. Plot Actual vs. Fitted Values
plt.figure(figsize=(10, 6))
plt.scatter(df["ln_price"], df["ln_load"], alpha=0.5, label="Actual Data")
plt.plot(df["ln_price"], model.predict(), color="red", label="Fitted Model")
plt.xlabel("ln(Price)")
plt.ylabel("ln(Load)")
plt.title(f"Elasticity Model: 𝛼 = {model.params['ln_price']:.3f}")
plt.legend()
plt.grid()
plt.show()