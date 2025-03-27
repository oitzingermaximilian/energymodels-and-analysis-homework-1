#%% Import data prep. functions
from prepare_input_data import prepare_combined_data
import statsmodels.api as sm
import importlib
import prepare_input_data

importlib.reload(prepare_input_data)


# Beispiel-Dateipfade
demand_file = "data_assignement_1/hourly_load_profile_electricity_AT_2023.xlsx"
price_file = "data_assignement_1/preise2023.csv"
weather_file = "data_assignement_1/Wetterdaten_Basel_2023.csv"
import_export_file = "data_assignement_1/Import_Export_Data.xlsx"
power_gen_file = "data_assignement_1/power_gen.xlsx"

combined_data = prepare_combined_data(demand_file, price_file, weather_file, import_export_file, power_gen_file)

#Create Model

# Neue Modellformulierung
X = combined_data[['Tageszeit_cos', 'Stromerzeugung', 'Strompreis', 'Temperatur']]
X = sm.add_constant(X)
y = combined_data['Nachfrage']

# Regression durchführen
model = sm.OLS(y, X)
results = model.fit()

# Ergebnisse anzeigen
print(results.summary())


