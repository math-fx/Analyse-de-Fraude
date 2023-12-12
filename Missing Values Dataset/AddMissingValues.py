import pandas as pd
import numpy as np

# Charger les données
data = pd.read_csv('Synthetic Dataset/SyntheticDataset.csv')

# Simuler des valeurs manquantes dans le dataset
np.random.seed(42)  # Pour assurer la reproductibilité
mask = np.random.rand(*data.shape) < 0.1  # Créez un masque de valeurs True/False (10% de valeurs manquantes)
data_with_missing = data.mask(mask)

# Enregistrez le nouveau dataset avec des valeurs manquantes
data_with_missing.to_csv('Missing Values Dataset/MVDataset.csv', index=False)
