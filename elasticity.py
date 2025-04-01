import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

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

prices_eur = (prices/100) #€/kWh

load_kWh = load * 1000 #kwh

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot für die Lastprofile (load)
ax1.plot(load_kWh.index, load_kWh, label='Stromlast', color='lightblue')
ax1.set_title('Zeitlicher Verlauf der Stromlast (2023)')
ax1.set_ylabel('Stromlast [kWh]')
ax1.grid(False)
ax1.legend()

# Plot für die Strompreise (prices)
ax2.plot(prices_eur.index, prices_eur, label='Strompreis', color='orange')
ax2.set_title('Zeitlicher Verlauf der Strompreise (2023)')
ax2.set_xlabel('Zeit (Stunden)')
ax2.set_ylabel('Strompreis [€/kWh]')
ax2.grid(False)
ax2.legend()

plt.tight_layout()
plt.show()


# --- Function to Perform ADF Test ---
def adf_test(series, name):
    result = adfuller(series.dropna())  # Drop NA to avoid errors
    print(f"ADF Test for {name}:")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Critical Values: {result[4]}")
    if result[1] < 0.05:
        print(f"  ✅ {name} is stationary (reject H0)")
    else:
        print(f"  ❌ {name} is non-stationary (fail to reject H0)")
    print("-" * 50)

# --- Apply ADF Test on Price and Load ---
adf_test(prices_eur, "Electricity Price")
adf_test(load_kWh, "Electricity Load")

# --- If Non-Stationary: Take First Difference and Re-test ---
if adfuller(prices_eur.dropna())[1] >= 0.05:
    adf_test(prices_eur.diff().dropna(), "Differenced Electricity Price")

if adfuller(load_kWh.dropna())[1] >= 0.05:
    adf_test(load_kWh.diff().dropna(), "Differenced Electricity Load")


load_kWh = load_kWh.reset_index()  # Moves index to a column
prices_eur = prices_eur.reset_index()
df = pd.merge(load_kWh, prices_eur, left_index=True, right_index=True)

# Nur gültige Werte für log-log Regression (price > 0, load > 0)
df_filtered = df[(df["AT"] > 0) & (df["Value_ScaleTo100"] > 0)]

print(df_filtered)

plt.scatter(df_filtered["AT"], df_filtered["Value_ScaleTo100"], alpha=0.5, label="Daten")
plt.xlabel("Preis [€/kWh]")
plt.ylabel("Nachfrage [kWh/h]")
plt.legend()
plt.show()


# Log-Transformation
log_price = np.log(df_filtered["AT"])
print(log_price)

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
plt.xlabel("log(Preis [€/MWh])")
plt.ylabel("log(Nachfrage [MWh/h])")
plt.legend()
plt.show()
