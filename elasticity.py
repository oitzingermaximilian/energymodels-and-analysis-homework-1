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


# For non-log methods, we can keep negatives but add small offset to avoid division by zero
load_clean = load.copy()
price_clean = prices.copy()
load_clean = load_clean + 1e-6  # Small offset to avoid exact zeros
price_clean = (price_clean + 1e-6) * 10 #to convert to €/Mwh

log_price = np.log(price_clean)
log_load = np.log(load_clean)