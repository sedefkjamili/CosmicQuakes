
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def sta_lta_filter(signal, sta_len=10, lta_len=100):
    """
    STA/LTA oranını hesaplar.
    
    :param signal: Zaman serisi verisi (1D NumPy array)
    :param sta_len: Kısa dönem penceresi uzunluğu (örneğin 10 örnek)
    :param lta_len: Uzun dönem penceresi uzunluğu (örneğin 100 örnek)
    :return: STA/LTA oranı dizisi
    """
    if len(signal) < max(sta_len, lta_len):
        raise ValueError(f"Veri uzunluğu ({len(signal)}) STA/LTA hesaplaması için yeterli değil. "
                         f"Minimum veri uzunluğu {max(sta_len, lta_len)} olmalıdır.")
    
    # Mutlak değer kullanarak sinyal enerjisini hesapla
    abs_signal = np.abs(signal)

    # STA ve LTA için kaydırmalı pencere ortalamaları
    sta = np.convolve(abs_signal, np.ones(sta_len) / sta_len, mode='valid')
    lta = np.convolve(abs_signal, np.ones(lta_len) / lta_len, mode='valid')

    # STA ve LTA dizilerini hizalama
    min_len = min(len(sta), len(lta))
    sta = sta[:min_len]
    lta = lta[:min_len]

    # Sıfıra bölmeyi önlemek için epsilon ekleyerek oranı hesapla
    epsilon = np.max(lta) * 1e-6  # Dinamik epsilon
    sta_lta_ratio = sta / (lta + epsilon)

    # Sonucu orijinal sinyal uzunluğuna döndür
    sta_lta_ratio = np.concatenate((np.zeros(lta_len - 1), sta_lta_ratio))

    return sta_lta_ratio

def prepare_data_for_cnn(signal, sta_len=10, lta_len=100, window_size=100):
    sta_lta_ratio = sta_lta_filter(signal, sta_len, lta_len)
    
    # 1D sinyali, CNN için uygun 2D formata dönüştürme
    segments = []
    for i in range(len(sta_lta_ratio) - window_size + 1):
        segment = sta_lta_ratio[i:i + window_size]
        segments.append(segment)

    # Segmentleri numpy array'e dönüştür
    X_input = np.array(segments)

    # Girişi 2D yap: (num_samples, window_size, 1)
    X_input = X_input[..., np.newaxis]
    
    return X_input

# Basit bir CNN modeli oluşturma
def create_cnn_model(window_size=100):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(window_size, 1)),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # İkili sınıflama örneği
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Örnek bir sinyal verisi
    signal = np.random.randn(1000)  # Örnek rastgele sinyal
    X_input = prepare_data_for_cnn(signal)
    
    # Modeli oluştur
    model = create_cnn_model(window_size=100)
    
    # Modeli özetle
    model.summary()
