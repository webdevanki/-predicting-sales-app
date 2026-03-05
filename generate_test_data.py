import pandas as pd
import numpy as np

np.random.seed(42)
n = 150

klienci = [
    'Kowalski Sp. z o.o.', 'Nowak Trading', 'TechVision S.A.', 'BudMat Polska',
    'AgriPol Sp. z o.o.', 'LogiTrans', 'MedSupply', 'RetailPro S.A.',
    'GreenEnergy Sp. z o.o.', 'AutoParts Polska', 'FoodDist S.A.', 'PrintMaster',
    'ChemTech Sp. z o.o.', 'SteelWork S.A.', 'EduTech Sp. z o.o.'
]

sprzedawcy = [
    'Anna Wiśniewska', 'Piotr Kowalczyk', 'Marta Jabłońska',
    'Tomasz Nowak', 'Katarzyna Wróbel'
]

branże = [
    'IT / Technologia', 'Budownictwo', 'Rolnictwo', 'Logistyka', 'Medycyna',
    'Retail', 'Energia', 'Motoryzacja', 'Spożywczy', 'Edukacja'
]

daty = pd.date_range('2023-01-01', '2024-06-30', periods=n)

liczba_produktow = np.random.randint(1, 50, n)
cena_jednostkowa = np.random.uniform(20, 500, n)
branza_arr = np.random.choice(branże, n)

# Wartość płatności zależy od liczby produktów, ceny i branży + szum
base = liczba_produktow * cena_jednostkowa
branza_mult = {
    'IT / Technologia': 1.3, 'Medycyna': 1.4, 'Energia': 1.2,
    'Budownictwo': 1.1, 'Rolnictwo': 0.9, 'Logistyka': 1.0,
    'Retail': 0.95, 'Motoryzacja': 1.15, 'Spożywczy': 0.85, 'Edukacja': 0.8
}
mult = np.array([branza_mult[b] for b in branza_arr])
zaplac = (base * mult + np.random.normal(0, 200, n)).clip(50).round(2)

df = pd.DataFrame({
    'ID': range(1001, 1001 + n),
    'Data zamowienia': daty.strftime('%Y-%m-%d'),
    'Nazwa klienta': np.random.choice(klienci, n),
    'Sprzedawca': np.random.choice(sprzedawcy, n),
    'Branża': branza_arr,
    'Liczba produktow': liczba_produktow,
    'Wartosc jednostkowa': cena_jednostkowa.round(2),
    'Zapłacono': zaplac,
    'Komentarz': np.random.choice(['', 'Pilne', 'Klient VIP', 'Reklamacja', ''], n)
})

df.to_csv('zamowienia_testowe.csv', index=False, encoding='utf-8')
print(f"Wygenerowano {len(df)} rekordów -> zamowienia_testowe.csv")
print(df.head(5).to_string())
