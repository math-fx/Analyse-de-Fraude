import pandas as pd
import random

# Charger le dataset
df = pd.read_csv('Large Dataset/CleanLargeDataset.csv')

# Définir le pourcentage de lignes à conserver (0.1% dans ce cas)
percentage_to_keep = 0.001

# Calculer le nombre de lignes à conserver
num_rows_to_keep = int(len(df) * percentage_to_keep)

# Sélectionner au hasard les indices des lignes à conserver
rows_to_keep = random.sample(range(len(df)), num_rows_to_keep)

# Créer un nouveau DataFrame avec les lignes sélectionnées au hasard
df_sampled = df.iloc[rows_to_keep]

# Enregistrer le nouveau DataFrame dans un fichier CSV
df_sampled.to_csv('Synthetic Dataset/SyntheticDataset.csv', index=False)

print(f"Le nouveau dataset a été enregistré avec {num_rows_to_keep} lignes au hasard sur un total de {len(df)} lignes du fichier d'origine.")
