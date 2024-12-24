import matplotlib.pyplot as plt
from src.preprocessing import load_data, process_data
from src.sta_lta import sta_lta_filter
import sys
import os

# Proje kök dizinini PYTHONPATH'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_data, process_data


# Veriyi yükle ve işleme
data = load_data('data/moon_data.csv')
data = process_data(data)

# STA/LTA hesaplama
sta_lta_ratio = sta_lta_filter(data['velocity'].values)


# STA/LTA oranını görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(data['time_abs'][1000:], sta_lta_ratio, label='STA/LTA Oranı')
plt.title('STA/LTA Oranı')
plt.xlabel('Zaman')
plt.ylabel('STA/LTA Oranı')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.show()
