import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def generate_spectrogram(signal_data, fs, nperseg=256, noverlap=128, save_path=None):
    """
    Sinyalin spektrogramını oluşturur ve kaydeder.

    :param signal_data: 1D NumPy array (Zaman serisi sinyali)
    :param fs: Örnekleme frekansı (Hz)
    :param nperseg: Pencere uzunluğu (default: 256)
    :param noverlap: Pencereler arası örtüşme miktarı (default: 128)
    :param save_path: Eğer belirtilirse, spektrogramı görüntü olarak kaydeder
    :return: Spektrogram verisi (frekans, zaman, güç yoğunluğu)
    """
    # STFT ile spektrogram hesapla
    f, t, Sxx = signal.spectrogram(signal_data, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Logaritmik dönüşüm (daha net görüntü için)
    Sxx_log = 10 * np.log10(Sxx + 1e-10)  # Log ölçeğe çevir (sıfırdan kaçınmak için 1e-10 eklenir)

    # Spektrogramı çiz
    plt.figure(figsize=(10, 6))
    plt.imshow(Sxx_log, aspect='auto', origin='lower', cmap='inferno',
               extent=[t.min(), t.max(), f.min(), f.max()])
    plt.colorbar(label='Güç Yoğunluğu (dB)')
    plt.xlabel('Zaman (s)')
    plt.ylabel('Frekans (Hz)')
    plt.title('Sinyal Spektrogramı')

    # Kaydet veya göster
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spektrogram kaydedildi: {save_path}")
    else:
        plt.show()

    return Sxx_log  # CNN için kullanılacak veri
