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


load = read_load_profiles(
    "data_assignement_1/hourly_load_profile_electricity_AT_2023.xlsx"
)
prices = read_hourly_prices("data_assignement_1/preise2023.csv")

prices_eur = prices * 10

load_kWh = load #* 1000

load_kWh = load_kWh.reset_index()  # Moves index to a column
prices_eur = prices_eur.reset_index()
df = pd.merge(load_kWh, prices_eur, left_index=True, right_index=True)

# Nur gültige Werte für log-log Regression (price > 0, load > 0)
df_filtered = df[(df["AT"] > 0) & (df["Value_ScaleTo100"] > 0)]

print(df_filtered)

# Log-Transformation
log_price = np.log(df_filtered["AT"])
log_load = np.log(df_filtered["Value_ScaleTo100"])


# Regression: log(load) ~ log(price)
X = sm.add_constant(log_price)  # Fügt den Intercept (log(C)) hinzu
y = log_load
model = sm.OLS(y, X).fit()
print(model.summary())
electricity_demand_elasticity = model.params['AT']
print('')
print("+ ESTIMATED ELECTRICITY DEMAND ELASTICITY: {:.4f}".format(electricity_demand_elasticity))
print('')
print(model.params)



# Ergebnisse
alpha = model.params["AT"]  # Elastizitätskoeffizient
p_value = model.pvalues["AT"]  # p-Wert für alpha
r_squared = model.rsquared  # Bestimmtheitsmaß

print(f"Elastizität (α): {alpha:.4f}")
print(f"p-Wert: {p_value:.4f}")
print(f"R²: {r_squared:.4f}")

# Signifikanzprüfung
if p_value < 0.05:
    print("→ Die Elastizität ist statistisch signifikant (p < 0.05).")
else:
    print("→ Die Elastizität ist NICHT signifikant.")

# Plot der Regression
plt.scatter(log_price, log_load, alpha=0.5, label="Daten")
plt.plot(log_price, model.fittedvalues, "r-", label="Regressionsgerade")
plt.xlabel("log(Preis)")
plt.ylabel("log(Nachfrage)")
plt.legend()
plt.show()
