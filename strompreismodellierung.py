import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Daten laden
load_data = pd.read_excel("hourly_load_profile_electricity_AT_2023.xlsx")
price_data = pd.read_csv("preise2023.csv", sep=";")

# Zeitvariablen extrahieren
load_data['DateUTC'] = pd.to_datetime(load_data['DateUTC'])
load_data['Stunde'] = load_data['DateUTC'].dt.hour
load_data['Wochentag'] = load_data['DateUTC'].dt.weekday
load_data['Monat'] = load_data['DateUTC'].dt.month
load_data.rename(columns={'Value': 'Last'}, inplace=True)
load_data['Preis'] = price_data['AT'].astype(float)

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
