from prepare_input_data import prepare_combined_data
import statsmodels.api as sm
import importlib
import prepare_input_data
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

importlib.reload(prepare_input_data)

# Beispiel-Dateipfade
demand_file = "data_assignement_1/hourly_load_profile_electricity_AT_2023.xlsx"
price_file = "data_assignement_1/preise2023.csv"
weather_file = "data_assignement_1/Wetterdaten_Basel_2023.csv"
import_export_file = "data_assignement_1/Import_Export_Data.xlsx"
power_gen_file = "data_assignement_1/power_gen.xlsx"

# Daten laden
combined_data = prepare_combined_data(demand_file, price_file, weather_file, import_export_file, power_gen_file)

# 1. Lag-Variable erstellen
combined_data = combined_data.copy()  # Explizite Kopie erstellen
combined_data.loc[:, "Nachfrage_lag1"] = combined_data["Nachfrage"].shift(1)
mean_lag = combined_data["Nachfrage"].mean()
combined_data.loc[:, "Nachfrage_lag1"] = combined_data["Nachfrage_lag1"].fillna(mean_lag)

# Tageszeit als kategorische Variable (6-Stunden-Blöcke)
combined_data['Tagesblock'] = pd.cut(combined_data['Tageszeit'],
                                   bins=[0, 6, 12, 18, 24],
                                   labels=['Nacht', 'Morgen', 'Nachmittag', 'Abend'])

# Dummy-Variablen mit 0/1-Kodierung
dummies = pd.get_dummies(combined_data['Tagesblock'],
                        prefix='Tageszeit',
                        drop_first=True).astype(int)

# Originale Tageszeit-Spalten entfernen (falls gewünscht)
combined_data.drop('Tagesblock', axis=1, inplace=True)

# Neue Variablen hinzufügen
combined_data = pd.concat([combined_data, dummies], axis=1)


#%%
# Dynamik Modell
X = combined_data[['Tageszeit_Morgen',
                  'Tageszeit_Nachmittag',
                  'Tageszeit_Abend', 'Strompreis']]
X = sm.add_constant(X)  # Konstante hinzufügen
y = combined_data['Nachfrage']

# 4. Regression
model = sm.OLS(y, X)
results = model.fit()

# 5. Ergebnisse + Diagnostik
print(results.summary())


#%% Multikollinearität prüfen (Dynamik Modell)

# Unabhängige Variablen (inklusive Konstante)
X = combined_data[['Tageszeit_Morgen',
                  'Tageszeit_Nachmittag',
                  'Tageszeit_Abend']]  # Falls du mehr Variablen hast, ergänzen!
X = sm.add_constant(X)  # Konstante für das Modell

# VIF berechnen
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Ergebnisse anzeigen
print(vif_data)


#%% Markt Modell
X2 = combined_data[['Stromerzeugung', 'Stromexport', 'Stromimport', 'Tageszeit_sin']]
X2 = sm.add_constant(X2)  # Konstante hinzufügen

# Regression durchführen
model2 = sm.OLS(y, X2)
results2 = model2.fit()

# Ergebnisse anzeigen
print("\nModell 2: Marktmodell")
print(results2.summary())

#%% Multikollinearität prüfen (Markt Modell)

# Unabhängige Variablen (inklusive Konstante)
X = combined_data[['Stromerzeugung', 'Stromexport', 'Stromimport', 'Tageszeit_sin']]  # Falls du mehr Variablen hast, ergänzen!
X = sm.add_constant(X)  # Konstante für das Modell

# VIF berechnen
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Ergebnisse anzeigen
print(vif_data)