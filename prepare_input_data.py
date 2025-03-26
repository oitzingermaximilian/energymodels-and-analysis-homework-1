#%%
import pandas as pd
import numpy as np

def convert_prices(input_file, price_column="AT"):
    """
    Liest eine CSV-Datei mit stündlichen Strompreisen in ct/kWh ein,
    rechnet sie in €/MWh um und gibt das DataFrame zurück.

    :param input_file: Pfad zur Eingabedatei (CSV)
    :param price_column: Name der Spalte mit den Strompreisen (default: "AT")
    :return: DataFrame mit der neuen Preisspalte "price_EUR_MWh"
    """
    df = pd.read_csv(input_file)

    if price_column not in df.columns:
        raise ValueError(f"Spalte '{price_column}' nicht in der CSV-Datei gefunden.")

    # Umrechnung von ct/kWh in €/MWh
    df["price_EUR_MWh"] = df[price_column] * 10
    return df


def load_import_export_data(file_path):
    """
    Lädt die Import-Export-Daten aus einer Excel-Datei und gibt sie als DataFrame zurück.

    :param file_path: Pfad zur Excel-Datei
    :return: DataFrame mit den Import-Export-Daten
    """
    df_import_export = pd.read_excel(file_path)

    # Überprüfen, ob die Spalten 'Stromexport' und 'Stromimport' existieren
    if "Stromexport" not in df_import_export.columns or "Stromimport" not in df_import_export.columns:
        raise ValueError("Die Excel-Datei muss die Spalten 'Stromexport' und 'Stromimport' enthalten.")

    # Sicherstellen, dass die Daten 8760 Stunden umfassen
    if len(df_import_export) != 8760:
        raise ValueError(f"Die Import-Export-Daten haben {len(df_import_export)} Zeilen, aber es werden genau 8760 erwartet.")

    return df_import_export


def load_demand_data(file_path):
    """
    Lädt die Verbrauchsdaten aus einer Excel-Datei und gibt sie als Series zurück.

    :param file_path: Pfad zur Excel-Datei
    :return: Series mit Verbrauchsdaten
    """
    rawdata = pd.read_excel(file_path)
    return rawdata["Value"]


def load_weather_data(file_path):
    """
    Lädt Wetterdaten aus einer CSV-Datei und gibt sie als DataFrame zurück.

    :param file_path: Pfad zur CSV-Datei
    :return: DataFrame mit den Wetterdaten
    """
    df_weather = pd.read_csv(file_path, skiprows=10)
    df_weather.columns = ["timestamp", "temperature"]
    df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"], format="%Y%m%dT%H%M")
    return df_weather


def prepare_combined_data(demand_file, price_file, weather_file, import_export_file):
    """
    Bereitet die kombinierten Daten aus den Dateien vor.

    :param demand_file: Pfad zur Excel-Datei mit den Verbrauchsdaten
    :param price_file: Pfad zur CSV-Datei mit den Preisdaten
    :param weather_file: Pfad zur CSV-Datei mit den Wetterdaten
    :param import_export_file: Pfad zur Excel-Datei mit den Import-Export-Daten
    :return: DataFrame mit den kombinierten Daten
    """
    data_demand = load_demand_data(demand_file)
    data_price = convert_prices(price_file)["price_EUR_MWh"]
    df_weather = load_weather_data(weather_file)
    df_import_export = load_import_export_data(import_export_file)

    # Überprüfen der Länge jeder Datei
    if len(data_demand) != 8760:
        raise ValueError(f"Die Verbrauchsdaten (Demand) haben {len(data_demand)} Zeilen, aber es werden genau 8760 erwartet.")
    if len(data_price) != 8760:
        raise ValueError(f"Die Preisdaten (Price) haben {len(data_price)} Zeilen, aber es werden genau 8760 erwartet.")
    if len(df_weather) != 8760:
        raise ValueError(f"Die Wetterdaten (Weather) haben {len(df_weather)} Zeilen, aber es werden genau 8760 erwartet.")

    # Erstellen der Tageszeit-Spalte
    tageszeiten = [i % 24 for i in range(8760)]  # Wiederholt 0-23 für 8760 Stunden

    # Kombinieren der Daten
    combined_data = pd.DataFrame({
        "Strompreis": data_price,
        "Nachfrage": data_demand,
        "Tageszeit": tageszeiten,
        "Temperatur": df_weather["temperature"],
        "Stromexport": df_import_export["Stromexport"],
        "Stromimport": df_import_export["Stromimport"]
    })

    # Hinzufügen der Sinus- und Cosinus-Spalten für die Tageszeit
    combined_data["Tageszeit_sin"] = np.sin(2 * np.pi * combined_data["Tageszeit"] / 24)
    combined_data["Tageszeit_cos"] = np.cos(2 * np.pi * combined_data["Tageszeit"] / 24)

    return combined_data





