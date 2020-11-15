import pandas as pd

def dataloader(filename):
    df = pd.read_csv("/content/drive/Shared drives/_Research Repository - Kunal Joshi/Data/Dataset.csv")
    print("Shape of dataframer:", df.shape)

    sensor_columns = ['sensor_{}'.format(i + 1) for i in range(50)]
    metadata = ["start_MP", "end_MP", "MP", "Run"]
    labels = ["label"]

    df.columns = sensor_columns + metadata + labels
    print(df.head())
    print(df.describe())
    return df