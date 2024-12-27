import numpy as np

def sta_lta_filter(signal, sta_len=10, lta_len=100):
    """
    STA/LTA oranını hesaplar.
    :param signal: Zaman serisi verisi
    :param sta_len: Kısa dönem penceresi uzunluğu
    :param lta_len: Uzun dönem penceresi uzunluğu
    :return: STA/LTA oranı
    """
    # Veri uzunluğu kontrolü
    if len(signal) < max(sta_len, lta_len):
        raise ValueError(f"Veri uzunluğu ({len(signal)}) STA/LTA hesaplaması için yeterli değil. "
                         f"Minimum veri uzunluğu {max(sta_len, lta_len)} olmalıdır.")
    
    # STA ve LTA hesaplama
    sta = np.cumsum(signal**2)
    lta = np.cumsum(signal**2)
    
    # STA ve LTA dizilerini uygun şekilde güncelleme
    sta[sta_len:] -= sta[:-sta_len]
    lta[lta_len:] -= lta[:-lta_len]
    
    # Boyutları eşitleme: STA ve LTA dizilerini sinyalle aynı uzunluğa getirme
    sta = sta[sta_len-1:]
    lta = lta[lta_len-1:]
    
    # STA/LTA oranını hesapla
    sta_lta_ratio = np.zeros(len(signal))  # Hedef sinyalin uzunluğunda bir dizi oluştur
    sta_lta_ratio[:len(lta)] = sta[:len(lta)] / lta  # Boyutları eşitleme
    
    # STA ve LTA dizilerinin boyutlarını eşitle
    if len(sta_lta_ratio) != len(signal):
        # STA/LTA oranı ile sinyalin uzunluğunu eşitle
        sta_lta_ratio = sta_lta_ratio[:len(signal)]
    
    # Sonuçları kontrol et
    print(f"STA/LTA oranının uzunluğu: {len(sta_lta_ratio)}")
    return sta_lta_ratio
