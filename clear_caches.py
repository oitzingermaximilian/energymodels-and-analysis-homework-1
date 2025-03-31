#%%
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

combined_data = prepare_combined_data(demand_file, price_file, weather_file, import_export_file, power_gen_file)