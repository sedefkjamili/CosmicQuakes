import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['time_abs', 'time_rel', 'velocity']
    return df

def process_data(df):
    df['time_abs'] = pd.to_datetime(df['time_abs'], format='%Y-%m-%dT%H:%M:%S.%f')
    missing_data = df.isnull().sum()
    print("Eksik Veriler:\n", missing_data)
    return df
