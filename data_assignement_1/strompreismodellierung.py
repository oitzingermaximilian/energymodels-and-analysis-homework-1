import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Daten laden
load_data = pd.read_excel("data_assignement_1/hourly_load_profile_electricity_AT_2023.xlsx")
price_data = pd.read_csv("data_assignement_1/preise2023.csv", sep=";")

# Zeitvariablen extrahieren
load_data['DateUTC'] = pd.to_datetime(load_data['DateUTC'])
load_data['Stunde'] = load_data['DateUTC'].dt.hour
load_data['Wochentag'] = load_data['DateUTC'].dt.weekday
load_data['Monat'] = load_data['DateUTC'].dt.month
load_data.rename(columns={'Value': 'Last'}, inplace=True)
load_data['Preis'] = price_data['AT'].astype(float)



# Annahme: Ihr DataFrame heißt 'load_data'
df = load_data[['Stunde', 'Wochentag', 'Monat', 'Last', 'Preis']].copy()

# Berechne die Schwellenwerte
upper_threshold = df['Preis'].quantile(0.99)  # Oberstes 1%
lower_threshold = df['Preis'].quantile(0.01)  # Unterstes 1%

# Filtere die extremen Werte
highest_1p = df[df['Preis'] >= upper_threshold]  # Höchste 1%
lowest_1p = df[df['Preis'] <= lower_threshold]   # Niedrigste 1%

# Optional: Erstelle einen DataFrame ohne die Extremwerte
df_filtered = df[(df['Preis'] > lower_threshold) & (df['Preis'] < upper_threshold)]

# 2. Lineare Regression (wie im ursprünglichen Modell)
X = sm.add_constant(df_filtered[['Stunde', 'Wochentag', 'Monat', 'Last']])
y = df_filtered['Preis']
model = sm.OLS(y, X).fit()

# 3. Ergebnisse anzeigen
print(model.summary())




# Modell-Daten
df = load_data[['Stunde', 'Wochentag', 'Monat', 'Last', 'Preis']]
X = sm.add_constant(df[['Stunde', 'Wochentag', 'Monat', 'Last']])
y = df['Preis']
model = sm.OLS(y, X).fit()
print(model.summary())

# Residuen-Plot
residuals = model.resid
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=50, kde=True, color='skyblue')
plt.title("Histogramm der Residuen – Modell 1")
plt.xlabel("Residuen")
plt.ylabel("Häufigkeit")
plt.grid(True)
plt.tight_layout()
plt.savefig("residuals_model1.png")


X = df[['Stunde', 'Wochentag', 'Monat', 'Last']]

# VIF-Berechnung
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

fitted_values = model.fittedvalues
residuals = model.resid


# Plots
plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Angepasste Strompreise [€/MWh] (Strompreis Modell 1)')
plt.ylabel('Residuen des Strompreises [€/MWh]')
plt.title('Residuen vs. Angepasste Werte')
plt.savefig("D:/Energiemodelle und Analysen/energymodels-and-analysis-homework-1/plots/residuen_vs_vorhersagen_Strom_modell1.png")