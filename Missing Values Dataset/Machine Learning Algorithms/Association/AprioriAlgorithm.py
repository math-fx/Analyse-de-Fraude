import pandas as pd
from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import time

# Enregistrez le temps de début
start_time = time.time()

# Charger les données depuis le fichier CSV
df = pd.read_csv("Missing values Dataset/CleanMVDataset.csv")

df.replace(to_replace=['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], value=[4, 5, 2, 3, 1], inplace=True)

# Séparation des caractéristiques (X) et de la variable cible (y)
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prendre un échantillon aléatoire des données pour réduire la taille
df_sample = df.sample(frac=0.001, random_state=42)

# Séparation des caractéristiques (X) et de la variable cible (y)
X = df_sample.drop('isFraud', axis=1)
y = df_sample['isFraud']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir les données en transactions pour Apriori
transactions = X_train.apply(lambda row: row.dropna().tolist(), axis=1).tolist()
# Application de l'algorithme Apriori
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Affichage des itemsets fréquents
print("Itemsets Fréquents :")
print(frequent_itemsets)

# Génération des règles d'association
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Affichage des règles d'association
print("\nRègles d'Association :")
print(rules)

# Enregistrez le temps de fin
end_time = time.time()
# Calculez le temps d'exécution total
execution_time = end_time - start_time

print(f'\nTemps d\'exécution total: {execution_time} secondes')
