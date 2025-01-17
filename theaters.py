import json
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Ucitavanje nefiltriranih podataka iz JSON fajla
with open('pozorista_RS.json', 'r', encoding='utf-8') as file:
  json_data = json.load(file)

# Pretvaranje JSON podataka u pandas DataFrame
df = pd.json_normalize(json_data)
# Filtriranje podataka, potrebni su podaci o posjecenosti profesionalnih pozorista na republickom nivou
filtered_df = df[(df['IDVrPod'] == '2') & (df['IDPozorista'] == '1') & (df['IDTer'] == 'RS')]

# Uzimamo sledece podatke: sezona i broj posjetilaca
result_df = filtered_df[['IDSezona', 'vrednost']]

# Cuvanje podataka u JSON fajl
result_df.to_json('posjecenost_profesionalnih_pozorista_RS.json', orient='records', lines=False, indent=4, force_ascii=False)

# Pristup FireStore bazi podataka
cred = credentials.Certificate("")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Ucitavanje filtriranih podataka iz JSON fajla
with open('posjecenost_profesionalnih_pozorista_RS.json') as json_file:
  data = json.load(json_file)

# Dodavanje podataka u FireStore kolekciju
doc_ref = db.collection('posjecenost_pozorista').document('podaci')
doc_ref.set({'data': data})

# Citanje podataka iz FireStore kolekcije
doc = doc_ref.get()
data = doc.to_dict().get('data', [])

df = pd.DataFrame(data) # Konverzija podataka u DataFrame

df['vrednost'] = pd.to_numeric(df['vrednost']) # vrijednost treba da bude broj
df['IDSezona_numeric'] = pd.factorize(df['IDSezona'])[0]  # Konverzija u numericku vrijednost

# Podjela podataka
X = df[['IDSezona_numeric']]  # Sezona
Y = df['vrednost']    # Broj posjetilaca

# Linearna regresija
regression = LinearRegression()
regression.fit(X, Y)
df['Trend'] = regression.predict(X) # Trend

plt.figure(figsize=(12, 6))
plt.plot(df['IDSezona'], df['Trend'], linestyle='--', color='red', label='Trend line')

# Alokacija prostora za prediktivne vrijednosti i pretvaranje niza u vektor
future_seasons_allocation = np.arange(len(df), len(df) + 5).reshape(-1, 1) # reshape(-1, 1) pretvara niz u vektor
future_values = regression.predict(future_seasons_allocation) # Predikcija za narednih 5 sezona

future_seasons = ['2022/23', '2023/24', '2024/25', '2025/26', '2026/27']
future_df = pd.DataFrame({
  'IDSezona': future_seasons,
  'vrednost': future_values
})
result_df = pd.concat([df, future_df], ignore_index=True) # Konacni DataFrame

# Vizualizacija 
# plt.plot(result_df['IDSezona'], result_df['vrednost'], marker='o', linestyle='-', color='blue', label='Оригиналне и предиктивне вриједности')
plt.plot(df['IDSezona'], df['vrednost'], marker='o', linestyle='-', color='blue', label='Original values')
plt.plot(result_df['IDSezona'].iloc[-5:], result_df['vrednost'].iloc[-5:], marker='o', linestyle='-', color='green', label='Predictive values')
plt.title('Attendance of professional theaters in Serbia')
plt.xlabel('Season')
plt.ylabel('Number of visitors (thousands)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()












