from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class DataCreator:
    def __init__(self, path: str, date_cols: str, cols_to_drop: list[str]):
        self.df = pd.read_csv(path, parse_dates = date_cols).sort_values(by = "Date").reset_index(drop=True)
        self.df.drop(columns = cols_to_drop, inplace=True)
    
    @staticmethod
    def create_variables(df):
        
        ############ Reemplace los None por la lógica solicitada ##################################
        df["Month"] = None
        df["Week"] = None
        df["Day"] = None
        df["Hour"] = None
        df["Minute"] = None
        df["Second"] = None
        df.drop(columns = ["Date"],inplace = True)
        return df
    
    def create_sequences(self, input_data: pd.DataFrame, target_column: str, sequence_length: int):
        self.idx = input_data.columns.tolist().index(target_column)
        
        sequences = []
        data_size = len(input_data)
        ############ Cree su código a continuación ##################################
        
        
        return sequences

    @staticmethod
    def split_data(df, month):
        ############ Cree su código a continuación ##################################

        return df_train, df_test
    
    @staticmethod
    def scale(train, test):
        ############ Cree su código a continuación ##################################

        return train_sc, test_sc, sc
    
    def inverse_scale(self, data: np.array):
        return data*self.sc.scale_[self.idx] + self.sc.min_[self.idx]

        return self.sc.inverse_transform(data)
    def run(self, month, seq_len):
        df = self.create_variables(self.df)
        self.train, self.test = self.split_data(df, month = month)
        self.train, self.test, self.sc = self.scale(self.train, self.test)
        self.train_sequences = self.create_sequences(self.train, target_column = "Close", sequence_length=seq_len)
        self.val_sequences = self.create_sequences(self.test, target_column = "Close",sequence_length=seq_len)
        return self.train_sequences, self.val_sequences
        
    