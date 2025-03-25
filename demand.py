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

tageszeiten = [i % 24 for i in range(8760)]  # Wiederholt 0-23 für 8760 Stunden

# Füge beide DataFrames zusammen
# Angenommen, data_demand und data_price haben beide 8760 Zeilen
combined_data = pd.DataFrame(
    {"Strompreis": data_price, "Nachfrage": data_demand, "Tageszeit": tageszeiten}
)


# Funktion zur Einteilung der Tageszeiten
def categorize_time(hour):
    if 0 <= hour < 6:
        return "Nacht"
    elif 6 <= hour < 12:
        return "Morgen"
    elif 12 <= hour < 18:
        return "Tag"
    else:
        return "Abend"

# Dummy-Variablen erstellen
combined_data["Tageszeit_kategorie"] = combined_data["Tageszeit"].apply(categorize_time)
dummies = pd.get_dummies(combined_data["Tageszeit_kategorie"], drop_first=False)  # Eine Kategorie weglassen (Referenz)

# Dummies ins DataFrame integrieren
combined_data = pd.concat([combined_data, dummies], axis=1)

print(dummies.columns)  # Zeigt die Namen der erstellten Dummy-Variablen


# %%
# create model
# Modell mit Dummy-Variablen

X = combined_data[['Strompreis', 'Morgen', 'Nacht', 'Tag', 'Abend']]  # Abend ist die Referenz
X = sm.add_constant(X)  # Konstante hinzufügen
y = combined_data['Nachfrage']

# Regression durchführen
model = sm.OLS(y, X)
results = model.fit()

# Ergebnisse anzeigen
print(results.summary())



#%%
print(combined_data)

