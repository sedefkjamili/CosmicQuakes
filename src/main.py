import matplotlib.pyplot as plt
import numpy as np
from src.preprocessing import load_data, process_data
from src.sta_lta import sta_lta_filter
from src.bandpass_filtering import bandpass_filter
from src.emd import apply_emd, plot_imfs
from src.fft_analysis import fft_analysis  
import sys
import os
import pandas as pd

# Proje kök dizinini PYTHONPATH'e ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Veriyi yükle ve işleme
data = load_data('data/xa.s12.00.mhz.1970-01-19HR00_evid00002.csv')
data = process_data(data)

# Sinyal durumunu kontrol et
print("Sinyal uzunluğu:", len(data['velocity']))
print("Sinyalde eksik veya hatalı değerler:", data['velocity'].isnull().sum())
print("Sinyaldeki minimum değer:", data['velocity'].min())
print("Sinyaldeki maksimum değer:", data['velocity'].max())

# Örnekleme frekansını hesapla
time_diff = pd.to_datetime(data['time_abs']).diff().dt.total_seconds().mean()
fs = 1 / time_diff  # Örnekleme frekansı
print(f"Örnekleme Frekansı (fs): {fs:.2f} Hz")

# Bandpass Filtreleme için Nyquist'e uygun kesim frekansları
lowcut = 0.1 * fs / 2  # Nyquist'in %10'u
highcut = 0.9 * fs / 2  # Nyquist'in %90'ı

print(f"Lowcut: {lowcut:.2f} Hz, Highcut: {highcut:.2f} Hz")

# Bandpass filtresini uygula
filtered_velocity = bandpass_filter(data['velocity'].values, lowcut, highcut, fs)

# Filtrelenmiş sinyali görselleştir
plt.figure(figsize=(10, 6))
plt.plot(data['time_abs'], data['velocity'], label='Ham Sinyal', alpha=0.5)
plt.plot(data['time_abs'], filtered_velocity, label='Filtrelenmiş Sinyal', linewidth=2)
plt.title('Bandpass Filtreleme')
plt.xlabel('Zaman')
plt.ylabel('Hız')
plt.legend()
plt.grid(True)
plt.show()

# Empirik Mod Ayrıştırma (EMD) uygulaması
imfs = apply_emd(filtered_velocity)
plot_imfs(imfs, original_signal=filtered_velocity)

# IMF 1 için Fourier Analizi (FFT)
fft_analysis(imfs[0], fs, title="IMF 1'in Frekans Spektrumu (FFT)")    
# STA/LTA hesaplama 
try:
    sta_lta_ratio = sta_lta_filter(filtered_velocity, sta_len=10, lta_len=50)

    if np.any(np.isnan(sta_lta_ratio)) or np.any(np.isinf(sta_lta_ratio)):
        print("Hata: STA/LTA oranında geçersiz (NaN veya sonsuz) değerler var!")

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
    print(f"STA/LTA hesaplama hatası: {e}")
