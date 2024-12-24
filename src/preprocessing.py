import pandas as pd

def load_data(file_path):
    """
    CSV dosyasından veri yükler.
    :param file_path: CSV dosyasının yolu
    :return: DataFrame
    """
    return pd.read_csv(file_path)

def process_data(df):
    """
    Veriyi işlemeye başlar: 
    - Zaman sütunlarını doğru formatta dönüştür
    - Eksik verileri kontrol et
    :param df: Yüklü veri
    :return: İşlenmiş veri
    """
    # 'time_abs' sütununu datetime formatına dönüştür
    df['time_abs'] = pd.to_datetime(df['time_abs'])
    
    # Eksik verileri kontrol et
    missing_data = df.isnull().sum()
    print("Eksik Veriler:\n", missing_data)
    
    # Veriyi döndür
    return df
