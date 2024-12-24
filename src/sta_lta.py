import numpy as np

def sta_lta_filter(signal, sta_len=100, lta_len=1000):
    """
    STA/LTA oranını hesaplar.
    :param signal: Zaman serisi verisi
    :param sta_len: Kısa dönem penceresi uzunluğu
    :param lta_len: Uzun dönem penceresi uzunluğu
    :return: STA/LTA oranı
    """
    sta = np.cumsum(signal**2)
    lta = np.cumsum(signal**2)
    
    sta[sta_len:] -= sta[:-sta_len]
    lta[lta_len:] -= lta[:-lta_len]

    sta = sta[sta_len-1:]
    lta = lta[lta_len-1:]
    
    sta_lta_ratio = sta / lta
    return sta_lta_ratio
