import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Charger le dataset
df = pd.read_csv('Large Dataset/LargeDataset.csv')

# Supprimer les colonnes non nécessaires pour la détection de fraude
columns_to_drop = ['nameOrig', 'nameDest']
df = df.drop(columns=columns_to_drop)

# Supprimer les lignes avec des valeurs manquantes
df.dropna()

# Supprimer les valeurs dupliquées
df.drop_duplicates()

# Encoder les variables catégorielles (type) en utilisant One-Hot Encoding
df = pd.get_dummies(df, columns=['type'])

# Normaliser les colonnes numériques avec StandardScaler
numeric_columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Réduction de dimension avec PCA 
pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df[numeric_columns]), columns=['PC1', 'PC2'])

# Concaténer les composants principaux avec le reste du dataframe
df = pd.concat([df, df_pca], axis=1)

# Supprimer les colonnes originales après réduction de dimension
df = df.drop(columns=numeric_columns)

# Sauvegarder le dataset nettoyé
df.to_csv('Large Dataset/CleanLargeDataset.csv', index=False)
