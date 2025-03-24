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
prices_euro_per_mwh = prices * 10
# Beispiel: Lade die Daten (Passe den Pfad an)
# df = pd.read_csv('your_data.csv')

# Angenommene Spaltennamen: 'Price' (in ct/kWh), 'Demand' (in MWh/h)
# Falls die Namen anders sind, anpassen!

# Shift-Transformation durchführen
shift_value = abs(prices_euro_per_mwh.min()) + 100
prices_shifted = prices_euro_per_mwh + shift_value

# Logarithmieren
df_logp = np.log(prices_shifted)
df_logq = np.log(load)

# Regression durchführen
X = sm.add_constant(df_logp)  # Konstante für das Modell hinzufügen
Y = df_logq
model = sm.OLS(Y, X).fit()

# Ergebnisse ausgeben
print(model.summary())

# Elastizität = Regressionskoeffizient von log_P
elasticity = model.params['const']  # Hier ist der Koeffizient des Preises
print(f'Preis-Elastizität der Nachfrage: {elasticity:.4f}')
# p-Wert des Preis-Koeffizienten
p_value = model.pvalues['const']
print(f'p-Wert der Preiselastizität: {p_value:.4f}')

# Signifikanz prüfen
if p_value < 0.05:
    print("Die Preiselastizität ist signifikant.")
else:
    print("Die Preiselastizität ist nicht signifikant.")



# Visualisierung der Regression
plt.scatter(df_logp, df_logq, alpha=0.5, label='Daten')
plt.plot(df_logp, model.predict(X), color='red', label='Regressionslinie')
plt.xlabel('Log(Preis, verschoben)')
plt.ylabel('Log(Nachfrage)')
plt.legend()
plt.show()
