import numpy as np
import matplotlib.pyplot as plt

def fft_analysis(signal, fs, title="Fourier Transform"):
    """
    Sinyalin Fourier Dönüşümünü hesaplar ve görselleştirir.
    
    Args:
        signal (numpy.ndarray): Sinyal verisi.
        fs (float): Örnekleme frekansı.
        title (str): Grafiğin başlığı.
    """
    n = len(signal)
    freq = np.fft.rfftfreq(n, d=1/fs)  # Frekans ekseni
    fft_values = np.fft.rfft(signal)   # FFT hesaplama

    plt.figure(figsize=(10, 6))
    plt.plot(freq, np.abs(fft_values), label='Frekans Spektrumu')
    plt.title(title)
    plt.xlabel('Frekans (Hz)')
    plt.ylabel('Genlik')
    plt.grid(True)
    plt.legend()
    plt.show()
