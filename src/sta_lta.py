import numpy as np

def sta_lta_filter(signal, sta_len=10, lta_len=100):
    """
    STA/LTA oranını hesaplar.
    
    :param signal: Zaman serisi verisi (1D NumPy array)
    :param sta_len: Kısa dönem penceresi uzunluğu
    :param lta_len: Uzun dönem penceresi uzunluğu
    :return: STA/LTA oranı dizisi
    """
    # Veri uzunluğu kontrolü
    if len(signal) < max(sta_len, lta_len):
        raise ValueError(f"Veri uzunluğu ({len(signal)}) STA/LTA hesaplaması için yeterli değil. "
                         f"Minimum veri uzunluğu {max(sta_len, lta_len)} olmalıdır.")
    
    # STA ve LTA hesaplama için kare alma
    squared_signal = signal**2

    # Kümülâtif toplam ile hesaplama
    sta = np.cumsum(squared_signal)
    lta = np.cumsum(squared_signal)
    
    # STA ve LTA dizilerini uygun şekilde güncelleme
    sta[sta_len:] -= sta[:-sta_len]
    lta[lta_len:] -= lta[:-lta_len]
    
    # STA ve LTA dizilerinin başlangıç kısmını keserek boyutlarını eşitleme
    sta = sta[lta_len-1:]
    lta = lta[lta_len-1:]

    # Sıfıra bölme hatasını önlemek için küçük bir epsilon ekleme
    epsilon = 1e-10  
    sta_lta_ratio = sta / (lta + epsilon)

    # STA/LTA oranını sinyalin uzunluğuna eşitleme
    sta_lta_ratio = np.pad(sta_lta_ratio, (0, len(signal) - len(sta_lta_ratio)), 'constant', constant_values=0)

    # Sonuçları kontrol et
    print(f"STA/LTA oranının uzunluğu: {len(sta_lta_ratio)}")
    
    return sta_lta_ratio
