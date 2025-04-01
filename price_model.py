#%%
from prepare_input_data import prepare_combined_data
import statsmodels.api as sm
import importlib
import prepare_input_data
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

importlib.reload(prepare_input_data)

# Beispiel-Dateipfade
demand_file = "data_assignement_1/hourly_load_profile_electricity_AT_2023.xlsx"
price_file = "data_assignement_1/preise2023.csv"
weather_file = "data_assignement_1/Wetterdaten_Basel_2023.csv"
import_export_file = "data_assignement_1/Import_Export_Data.xlsx"
power_gen_file = "data_assignement_1/power_gen.xlsx"

combined_data = prepare_combined_data(demand_file, price_file, weather_file, import_export_file, power_gen_file)

# 1. Lag-Variable erstellen
combined_data = combined_data.copy()  # Explizite Kopie erstellen
combined_data.loc[:, "Nachfrage_lag1"] = combined_data["Nachfrage"].shift(1)
mean_lag = combined_data["Nachfrage"].mean()
combined_data.loc[:, "Nachfrage_lag1"] = combined_data["Nachfrage_lag1"].fillna(mean_lag)



# Lags des Strompreises erstellen
combined_data.loc[:, "Strompreis_lag1"] = combined_data["Strompreis"].shift(1)
combined_data.loc[:, "Strompreis_lag24"] = combined_data["Strompreis"].shift(24)
combined_data.loc[:, "Strompreis_lag168"] = combined_data["Strompreis"].shift(168)

# Fehlende Werte mit dem Mittelwert des Strompreises auffüllen
mean_price = combined_data["Strompreis"].mean()
combined_data.loc[:, ["Strompreis_lag1", "Strompreis_lag24", "Strompreis_lag168"]] = combined_data[
    ["Strompreis_lag1", "Strompreis_lag24", "Strompreis_lag168"]
].fillna(mean_price)


#%%
# herausfinden welcher lag den höchsten einfluss hat.
X = combined_data[['Strompreis_lag1', 'Strompreis_lag24', 'Strompreis_lag168']]
X = sm.add_constant(X)  # Konstante hinzufügen
y = combined_data['Strompreis']

# 4. Regression
model = sm.OLS(y, X)
results = model.fit()

# 5. Ergebnisse + Diagnostik
print(results.summary())

#%%
X = combined_data[['Nachfrage_lag1', 'Temperatur', 'Stromimport']]
X = sm.add_constant(X)  # Konstante hinzufügen
y = combined_data['Strompreis']

# 4. Regression
model = sm.OLS(y, X)
results = model.fit()

# 5. Ergebnisse + Diagnostik
print(results.summary())
