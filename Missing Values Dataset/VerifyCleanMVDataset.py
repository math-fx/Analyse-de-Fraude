import pandas as pd

def count_missing_values(file_path):
    # Charger le fichier CSV dans un DataFrame
    df = pd.read_csv(file_path)

    # Obtenir le nombre de valeurs manquantes par colonne
    missing_values = df.isnull().sum()

    # Afficher le nombre de valeurs manquantes par colonne
    print("Nombre de valeurs manquantes par colonne :")
    print(missing_values)

if __name__ == "__main__":
    # Remplacez "votre_fichier.csv" par le chemin de votre fichier CSV
    fichier_csv = "Missing Values Dataset/CleanMVDataset.csv"

    # Appeler la fonction pour compter les valeurs manquantes
    count_missing_values(fichier_csv)
