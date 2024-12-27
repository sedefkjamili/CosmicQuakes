import matplotlib.pyplot as plt
from src.preprocessing import load_data, process_data
from src.sta_lta import sta_lta_filter
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Veriyi yükle ve işleme
data = load_data('data/moon_data.csv')
data = process_data(data)

# Sinyal durumunu kontrol et
print("Sinyal uzunluğu:", len(data['velocity']))
print("Sinyalde eksik veya hatalı değerler:", data['velocity'].isnull().sum())
print("Sinyaldeki minimum değer:", data['velocity'].min())
print("Sinyaldeki maksimum değer:", data['velocity'].max())

# STA/LTA hesaplama
try:
    sta_lta_ratio = sta_lta_filter(data['velocity'].values, sta_len=1, lta_len=2)
    if len(sta_lta_ratio) > 0:
        time_abs_trimmed = data['time_abs'][-len(sta_lta_ratio):]
        plt.figure(figsize=(10, 6))
        plt.plot(time_abs_trimmed, sta_lta_ratio, label='STA/LTA Oranı')
        plt.title('STA/LTA Oranı')
        plt.xlabel('Zaman')
        plt.ylabel('STA/LTA Oranı')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("STA/LTA oranı hesaplanamadı. Lütfen veri uzunluğunu ve parametreleri kontrol edin.")
except ValueError as e:
    print(e)
