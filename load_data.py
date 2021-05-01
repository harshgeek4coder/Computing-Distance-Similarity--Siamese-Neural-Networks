import pandas as pd

def load_data():
    data_path='/content/drive/MyDrive/Siamese/data/Quora/train.csv'
    data=pd.read_csv(data_path)
    data_length=len(data)

    print("Data Loaded successfully..")

    return data,data_length