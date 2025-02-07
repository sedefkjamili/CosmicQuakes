from PyEMD import EMD
import matplotlib.pyplot as plt
import numpy as np

def apply_emd(signal):
    """
    Sinyale Empirical Mode Decomposition (EMD) uygular.
    
    Parameters:
        signal (numpy.ndarray): Girdi sinyali.

    Returns:
        list of numpy.ndarray: IMF'ler (Intrinsic Mode Functions).
    """
    emd = EMD()
    imfs = emd(signal)
    return imfs

def plot_imfs(imfs, original_signal=None):
    """
    IMF'leri ve opsiyonel olarak orijinal sinyali görselleştirir.
    
    Parameters:
        imfs (list of numpy.ndarray): IMF'ler.
        original_signal (numpy.ndarray): Opsiyonel olarak orijinal sinyal.
    """
    num_imfs = len(imfs)
    plt.figure(figsize=(10, 6))
    
    if original_signal is not None:
        plt.subplot(num_imfs + 1, 1, 1)
        plt.plot(original_signal, label='Orijinal Sinyal')
        plt.title('Orijinal Sinyal')
        plt.grid(True)
        plt.legend()
    
    for i, imf in enumerate(imfs):
        plt.subplot(num_imfs + 1, 1, i + 2)
        plt.plot(imf, label=f'IMF {i+1}')
        plt.title(f'IMF {i+1}')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show() 