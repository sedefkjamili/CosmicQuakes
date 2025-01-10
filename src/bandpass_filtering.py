import numpy as np
import scipy.signal as signal

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Bandpass filtresi uygular.
    :param data: Zaman serisi verisi (numpy array veya liste)
    :param lowcut: Alçak kesim frekansı (Hz)
    :param highcut: Yüksek kesim frekansı (Hz)
    :param fs: Örnekleme frekansı (Hz)
    :param order: Filtrenin derecesi
    :return: Filtrelenmiş veri
    """
    nyquist = 0.5 * fs  # Nyquist frekansı
    low = lowcut / nyquist
    high = highcut / nyquist

    # Bandpass filtresi tasarımı
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Filtreden geçir
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data

