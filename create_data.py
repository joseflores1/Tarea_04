from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataCreator:
    def __init__(self, path: str, date_cols: str, cols_to_drop: list[str]):
        self.df = pd.read_csv(path, parse_dates = date_cols).sort_values(by = "Date").reset_index(drop=True)
        self.df.drop(columns = cols_to_drop, inplace=True)
    
    @staticmethod
    def create_variables(df):
        """
        Crear variables estacionales para determinar patrones temporales.
        Asegurarse de que la columna 'Date' está en formato datetime.
        """
        # Verificar si 'Date' es de tipo datetime; si no, convertirla
        if not np.issubdtype(df['Date'].dtype, np.datetime64):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Extraer variables estacionales
        df["Month"] = df["Date"].dt.month
        df["Week"] = df["Date"].dt.isocalendar().week
        df["Day"] = df["Date"].dt.day
        df["Hour"] = df["Date"].dt.hour
        df["Minute"] = df["Date"].dt.minute
        df["Second"] = df["Date"].dt.second

        # Eliminar la columna 'Date' y "Unix"   sdespués de extraer las variables
        df.drop(columns=["Date"], inplace=True)
        df.drop(columns=["Unix"], inplace=True)
        return df

    
    def create_sequences(self, input_data: pd.DataFrame, target_column: str, sequence_length: int):
        """
        Generar datos secuenciales utilizando una sliding window.

        Args:
            input_data (pd.DataFrame): DataFrame con los datos.
            target_column (str): Columna objetivo a predecir.
            sequence_length (int): Longitud de la secuencia a generar.

        Returns:
            list: Lista de tuplas (secuencia, valor objetivo).
        """
        self.idx = input_data.columns.tolist().index(target_column)  # Índice de la columna objetivo
        
        sequences = []
        data_size = len(input_data)
        
        for i in range(data_size - sequence_length):
            # Crear una secuencia de longitud sequence_length
            seq = input_data.iloc[i:i + sequence_length].values
            # Asociar el valor objetivo del registro inmediatamente siguiente
            target = input_data.iloc[i + sequence_length][target_column]
            sequences.append((seq, target))
        
        return sequences


    @staticmethod
    def split_data(df, month):
        """
        Dividir los datos en conjuntos de entrenamiento y prueba (validación) según el mes dado.
        Los datos hasta el mes especificado (inclusive) son parte del entrenamiento,
        mientras que los datos posteriores son parte de la validación.

        Args:
            df (pd.DataFrame): DataFrame con los datos preprocesados.
            month (int): Mes límite para separar los datos (inclusive en train).

        Returns:
            tuple: DataFrame de entrenamiento y DataFrame de validación.
        """
        # Dividir en entrenamiento y prueba basándose en el mes
        df_train = df[df["Month"] <= month]  # Datos hasta el mes especificado inclusive
        df_test = df[df["Month"] > month]   # Datos después del mes especificado
        return df_train, df_test

    
    @staticmethod
    def scale(train, test):
        """
        Escalar los datos de entrenamiento y prueba usando MinMaxScaler.

        Args:
            train (pd.DataFrame): DataFrame del conjunto de entrenamiento.
            test (pd.DataFrame): DataFrame del conjunto de prueba.

        Returns:
            tuple: Datos escalados de entrenamiento, datos escalados de prueba y la instancia del MinMaxScaler.
        """
        sc = MinMaxScaler()  # Crear instancia de MinMaxScaler
        
        # Ajustar el escalador con los datos de entrenamiento y transformar ambos conjuntos
        train_sc = sc.fit_transform(train)
        test_sc = sc.transform(test)
    
        return train_sc, test_sc, sc

    
    def inverse_scale(self, data: np.array):
        #return data*self.sc.scale_[self.idx] + self.sc.min_[self.idx]
        return self.sc.inverse_transform(data)
    def run(self, month, seq_len):
        df = self.create_variables(self.df)
        self.train, self.test = self.split_data(df, month = month)
        self.train, self.test, self.sc = self.scale(self.train, self.test)
        self.train_sequences = self.create_sequences(self.train, target_column = "Close", sequence_length=seq_len)
        self.val_sequences = self.create_sequences(self.test, target_column = "Close",sequence_length=seq_len)
        return self.train_sequences, self.val_sequences
        
if __name__ == "__main__":
    # Ruta al archivo de datos
    path = "Binance_BTCUSDT_2024_minute.csv"
    date_cols = ["Date"]  # Columna que contiene la fecha
    cols_to_drop = ["Symbol"]  # Columnas a eliminar

    # Crear instancia de DataCreator
    data_creator = DataCreator(path=path, date_cols=date_cols, cols_to_drop=cols_to_drop)

    # Mes límite para dividir los datos y longitud de secuencia
    month = 10
    seq_len = 120

    # Paso 1: Crear variables estacionales
    print("Probando creación de variables...")
    df_with_variables = data_creator.create_variables(data_creator.df)
    print(f"Variables creadas: {df_with_variables.columns}")

    # Paso 2: Dividir datos en entrenamiento y validación
    print("\nDividiendo datos en entrenamiento y validación...")
    df_train, df_test = data_creator.split_data(df_with_variables, month=month)
    print(f"Tamaño del conjunto de entrenamiento: {df_train.shape}")
    print(f"Tamaño del conjunto de validación: {df_test.shape}")

    # Paso 3: Escalar datos
    print("\nEscalando datos de entrenamiento y validación...")
    train_scaled, test_scaled, scaler = data_creator.scale(df_train, df_test)
    print("Datos de entrenamiento escalados (primeras filas):")
    print(train_scaled[:5])

    # Paso 4: Crear secuencias a partir de los datos escalados
    print("\nGenerando secuencias de entrenamiento y validación...")
    train_sequences = data_creator.create_sequences(
        pd.DataFrame(train_scaled, columns=df_train.columns),
        target_column="Close",
        sequence_length=seq_len,
    )
    val_sequences = data_creator.create_sequences(
        pd.DataFrame(test_scaled, columns=df_test.columns),
        target_column="Close",
        sequence_length=seq_len,
    )
    print(f"Secuencias de entrenamiento generadas: {len(train_sequences)}")
    print(f"Secuencias de validación generadas: {len(val_sequences)}")
    
    # Confirmación final
    print("\nPruebas completadas en el orden del método 'run'.")
