import pandas as pd
import statsmodels.api as sm
import numpy as np

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
print(prices.head())  # Zeigt die ersten 5 Zeilen des prices DataFrames
print(load.head())    # Zeigt die ersten 5 Zeilen des load DataFrames

# Indizes setzen (0 bis 8759 für beide DataFrames)
prices.index = range(8760)
load.index = range(8760)

# DataFrames zusammenführen (auf Basis des Index)
df = pd.concat([prices, load], axis=1)

df['AT'] = df['AT'] * 10

# Überprüfen auf NaN oder unendliche Werte
if df['AT'].isnull().any() or np.any(np.isinf(df['AT'])):
    print("Fehler: Es gibt NaN oder unendliche Werte in den Preisen.")

# Daten vor der logarithmischen Transformation bereinigen
df = df[df['AT'] > -1]  # Entferne alle Zeilen, bei denen der Preis <= -1 ist (damit log(x+1) funktioniert)

# Logarithmische Transformation der Preise und Nachfrage
df['log_price'] = np.log(df['AT'] + 1)  # log(P+1), falls AT negative Werte hat
df['log_demand'] = np.log(df['Value_ScaleTo100'])

# Überprüfen auf NaN oder unendliche Werte nach der Transformation
if df['log_price'].isnull().any() or np.any(np.isinf(df['log_price'])):
    print("Fehler: Es gibt NaN oder unendliche Werte nach der Transformation.")

# Regression durchführen, wenn die Daten in Ordnung sind
X = sm.add_constant(df['log_price'])  # Preis (log-transformed)
Y = df['log_demand']  # Nachfrage (log-transformed)

# OLS-Regression durchführen
model = sm.OLS(Y, X).fit()

# Ergebnisse ausgeben
print(model.summary())

# Preiselastizität als der Koeffizient des Preises
price_elasticity = model.params['log_price']
print(f'Preis-Elastizität der Nachfrage: {price_elasticity:.4f}')

