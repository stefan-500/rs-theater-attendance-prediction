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

# Pretvaranje JSON podataka u DataFrame
df = pd.json_normalize(json_data)

# Filtriranje podataka, potrebni su podaci o posjecenosti pozorista na republickom nivou
filtered_df = df[(df['IDVrPod'] == '2') & (df['IDTer'] == 'RS')]
result_df = filtered_df[['IDSezona', 'vrednost']] # sezona i broj posjetilaca
result_df = result_df.reset_index()

# Za svaku sezonu izracunati i sacuvati vrijednost (za svaku sezonu postoje po 3 reda u result_df)
new_df = pd.DataFrame()
rows = []
iteration = 0

for unique_season in result_df['IDSezona'].unique():    # DataFrame bez duplikata
  vrednost = 0
  for index, item in result_df.iterrows():              # DataFrame sa duplikatima
    if item['IDSezona'] == unique_season:
      iteration += 1
      item['vrednost'] = pd.to_numeric(item['vrednost'])
      vrednost += item['vrednost']
      if iteration == 3:
        rows.append({'IDSezona': item['IDSezona'], 'vrednost': vrednost})
        iteration = 0        
        # Brisu se redovi uracunate sezone da bi se smanjio broj iteracija
        result_df.drop([index, index-1, index-2], inplace=True)
        break
    else:
      continue
rows_df = pd.DataFrame.from_records(rows)
new_df = pd.concat([new_df, rows_df])

# Cuvanje podataka u JSON fajl
new_df.to_json('posjecenost_pozorista_RS.json', orient='records', lines=False, indent=4, force_ascii=False)

# Pristup FireStore bazi podataka
cred = credentials.Certificate("")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Ucitavanje filtriranih podataka iz JSON fajla
with open('posjecenost_pozorista_RS.json') as json_file:
  data = json.load(json_file)

# Dodavanje podataka u FireStore kolekciju
doc_ref = db.collection('posjecenost_pozorista').document('podaci')
doc_ref.set({'data': data})

# Citanje podataka iz FireStore kolekcije
doc = doc_ref.get()
data = doc.to_dict().get('data', [])

df = pd.DataFrame(data)
df['vrednost'] = pd.to_numeric(df['vrednost']) # vrijednost treba da bude broj
df['IDSezona_numeric'] = pd.factorize(df['IDSezona'])[0]  # konverzija u numericku vrijednost

# Podjela podataka
X = df[['IDSezona_numeric']]
Y = df['vrednost']

# Linearna regresija
regression = LinearRegression()
X = X.values
regression.fit(X, Y)
df['Trend'] = regression.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(df['IDSezona'], df['Trend'], linestyle='--', color='red', label='Trend line')

# Alokacija prostora za prediktivne vrijednosti i pretvaranje niza u vektor
future_seasons_allocation = np.arange(len(df), len(df) + 5).reshape(-1, 1) # reshape(-1, 1) pretvara niz u vektor
future_values = regression.predict(future_seasons_allocation) # predikcija za narednih 5 sezona

future_seasons = ['2022/23', '2023/24', '2024/25', '2025/26', '2026/27']
future_df = pd.DataFrame({
  'IDSezona': future_seasons,
  'vrednost': future_values
})
result_df = pd.concat([df, future_df], ignore_index=True) # konacni DataFrame

plt.plot(df['IDSezona'], df['vrednost'], marker='o', linestyle='-', color='blue', label='Original values')
plt.plot(result_df['IDSezona'].iloc[-5:], result_df['vrednost'].iloc[-5:], marker='o', linestyle='-', color='green', label='Predictive values')
plt.title('Attendance of theaters in Serbia')
plt.xlabel('Season')
plt.ylabel('Number of visitors (thousands)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()