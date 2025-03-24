import pandas as pd
import os

os.chdir(
    "D:\Energiemodelle und Analysen\energymodels-and-analysis-homework-1\data_assignement_1"
)
# choose your path


# prepare input data


def convert_prices(input_file, price_column="AT"):
    """
    Liest eine CSV-Datei mit stündlichen Strompreisen in ct/kWh ein,
    rechnet sie in €/MWh um und speichert die neue Datei.

    :param input_file: Pfad zur Eingabedatei (CSV)
    :param output_file: Pfad zur Ausgabedatei (CSV)
    :param price_column: Name der Spalte mit den Strompreisen (default: "price_ct_kWh")
    """
    # CSV einlesen
    df = pd.read_csv(input_file)

    # Sicherstellen, dass die Preisspalte existiert
    if price_column not in df.columns:
        raise ValueError(f"Spalte '{price_column}' nicht in der CSV-Datei gefunden.")

    # Umrechnung von ct/kWh in €/MWh
    df["price_EUR_MWh"] = df[price_column] * 10

    print("Umrechnung abgeschlossen.")
    return df


rawdata = pd.read_excel("hourly_load_profile_electricity_AT_2023.xlsx")
data_demand = rawdata["Value"]

data_price = convert_prices("preise2023.csv")
data_price = data_price["price_EUR_MWh"]

tageszeiten = [i % 24 for i in range(8760)]  # Wiederholt 0-23 für 8760 Stunden

# Füge beide DataFrames zusammen
# Angenommen, data_demand und data_price haben beide 8760 Zeilen
combined_data = pd.DataFrame(
    {"Strompreis": data_price, "Nachfrage": data_demand, "Tageszeit": tageszeiten}
)

# %%
# create model

import statsmodels.api as sm

# Unabhängige Variablen (Strompreis und Tageszeit)
X = combined_data[["Strompreis", "Tageszeit"]]  # Strompreis und Tageszeit
X = sm.add_constant(X)  # Fügt den konstanten Term (Beta_0) hinzu

# Abhängige Variable (Nachfrage)
y = combined_data["Nachfrage"]

# Erstelle das lineare Regressionsmodell
model = sm.OLS(y, X)  # OLS (Ordinary Least Squares) Regressionsmodell
results = model.fit()  # Fitte das Modell

# Zeige die Ergebnisse der Regression an
print(results.summary())
