#%%

import pandas as pd
import statsmodels.api as sm
import numpy as np
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


rawdata = pd.read_excel("data_assignement_1/hourly_load_profile_electricity_AT_2023.xlsx")
data_demand = rawdata["Value"]

data_price = convert_prices("data_assignement_1/preise2023.csv")
data_price = data_price["price_EUR_MWh"]


# Datei einlesen und Header überspringen
file_path = "data_assignement_1/Wetterdaten_Basel_2023.csv"  # Pfad zur Datei anpassen

# Die Datei einlesen, wobei die ersten Zeilen (Metadaten) übersprungen werden
df_weather = pd.read_csv(file_path, skiprows=10)  # 10 Zeilen überspringen (anpassen, falls nötig)

# Spalten umbenennen (optional)
df_weather.columns = ["timestamp", "temperature"]

# Timestamp in ein datetime-Format umwandeln
df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"], format="%Y%m%dT%H%M")



tageszeiten = [i % 24 for i in range(8760)]  # Wiederholt 0-23 für 8760 Stunden
# Füge beide DataFrames zusammen
# Angenommen, data_demand und data_price haben beide 8760 Zeilen
combined_data = pd.DataFrame(
    {"Strompreis": data_price, "Nachfrage": data_demand, "Tageszeit": tageszeiten}
)

combined_data['Temperatur']=df_weather["temperature"]
combined_data["Tageszeit_sin"] = np.sin(2 * np.pi * combined_data["Tageszeit"] / 24)
combined_data["Tageszeit_cos"] = np.cos(2 * np.pi * combined_data["Tageszeit"] / 24)
#%%
print(combined_data)


X = combined_data[['Strompreis', 'Tageszeit_cos', 'Temperatur']]
X = sm.add_constant(X)
y = combined_data['Nachfrage']

model = sm.OLS(y, X)
results = model.fit()

print(results.summary())


