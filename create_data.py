from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

class DataCreator:
    def __init__(self, path: str, date_cols: str, cols_to_drop: list[str]) -> pd.DataFrame:
        self.df = pd.read_csv(path, parse_dates = date_cols).sort_values(by = "Date").reset_index(drop = True)
        self.df.drop(columns = cols_to_drop, inplace = True)
    
    @staticmethod
    def create_variables(df):

        df["Month"] = df["Date"].dt.month
        df["Week"] = df["Date"].dt.isocalendar().week
        df["Day"] = df["Date"].dt.day
        df["Hour"] = df["Date"].dt.hour
        df["Minute"] = df["Date"].dt.minute
        df["Second"] = df["Date"].dt.second


        df.drop(columns = ["Date"],inplace = True)
        return df
    
    def create_sequences(self, input_data: pd.DataFrame, target_column: str, sequence_length: int) -> list:

        """
        Genera secuencias utilizando una sliding window.
        
        Parámetros:
            input_data (pd.DataFrame): DataFrame de entrada con las características.
            target_column (str): Nombre de la columna objetivo a predecir.
            sequence_length (int): Largo de las secuencias a generar.
        
        Retorna:
            sequences (list): Lista de tuplas (X, y), donde:
                - X: Array NumPy con los valores de las secuencias (shape = [sequence_length, n_features]).
                - y: Valor de la columna objetivo inmediatamente después de la secuencia.
                
        """
        
        sequences = []

        data = input_data.to_numpy()

        data_size = len(data)

        target_idx = input_data.columns.get_loc(target_column)

        #We may take one less sequence because we get out of bounds trying to get the target... :
        for i in range(data_size - sequence_length):

            X = data[i : i + sequence_length, :]

            y = data[i + sequence_length, target_idx]

            sequences.append((X, y))

        return sequences


    @staticmethod
    def split_data(df, month: int) -> pd.DataFrame:

        """
        Separa los datos en train y test en función del mes dado.
        
        Parámetros:
            df (pd.DataFrame): El DataFrame procesado que contiene las variables extraídas.
            month (int): El número del mes (1 a 12) que determina el corte entre train y test.
        
        Retorna:
            df_train (pd.DataFrame): Subconjunto de datos que incluye hasta el mes dado.
            df_test (pd.DataFrame): Subconjunto de datos posterior al mes dado.
        """
        
        df_train = df[df["Month"] <= month]

        df_test = df[df["Month"] > month]

        return df_train, df_test
    
    @staticmethod
    def scale(train: pd.DataFrame, test: pd.DataFrame) -> tuple:

        """
        Escala los datos de train y test utilizando MinMaxScaler de Scikit-Learn.
        
        Parámetros:
            train (pd.DataFrame): Conjunto de entrenamiento.
            test (pd.DataFrame): Conjunto de prueba.
        
        Retorna:
            train_sc (pd.DataFrame): Conjunto de entrenamiento escalado.
            test_sc (pd.DataFrame): Conjunto de prueba escalado.
            sc (MinMaxScaler): Instancia del MinMaxScaler utilizada para escalar los datos.
        """

        sc = MinMaxScaler()

        train_sc = pd.DataFrame(sc.fit_transform(train), columns = train.columns, index = train.index)
        test_sc = pd.DataFrame(sc.transform(test), columns = test.columns, index = test.index)

        return train_sc, test_sc, sc
    
    def inverse_scale(self, predictions: list, target_column = "Close") -> list:
        """
        Desescala una lista de tensores de predicciones a sus valores originales.
        
        Parámetros:
            predictions (list): Lista de tensores con predicciones escaladas de dimensiones [batch_size, 1].
        
        Retorna:
            descaled_values (list): Lista continua de valores desescalados.
        """
        max_value = self.descaled_train[target_column].max()
        min_value = self.descaled_train[target_column].min()

        # Convertir el tensor de predicciones a un array NumPy
        predictions_numpy = predictions.squeeze(1).numpy()  # Quitar la dimensión extra y mover a CPU

        # Aplicar la fórmula de desescalado
        descaled_values = predictions_numpy * (max_value - min_value) + min_value

        return descaled_values.tolist()


    def run(self, month, seq_len, tc = "Close") :
        
        df = self.create_variables(self.df)
        self.train, self.test = self.split_data(df, month = month)

        self.descaled_train = self.train.copy()
        self.descaled_test = self.test.copy()
        
        self.train, self.test, self.sc = self.scale(self.train, self.test)

        self.train_sequences = self.create_sequences(self.train, target_column = tc, sequence_length = seq_len)
        self.val_sequences = self.create_sequences(self.test, target_column = tc ,sequence_length = seq_len)

        return self.train_sequences, self.val_sequences

    def plot_split_data(self, target_column: str = "Close", train_preds: list = None, test_preds: list = None):
        """
        Grafica los datos de entrenamiento, validación y, opcionalmente, las predicciones.

        Parámetros:
            target_column (str): Nombre de la columna objetivo.
            train_preds (list): Predicciones desescaladas del conjunto de entrenamiento (opcional).
            test_preds (list): Predicciones desescaladas del conjunto de validación (opcional).
        """
        # Crear el gráfico
        plt.figure(figsize = (15, 6))

        # Graficar datos de entrenamiento
        plt.plot(self.descaled_train.index, self.descaled_train[target_column], 
                label = f"Train ({target_column})", color = "blue", alpha = 0.7)

        # Graficar datos de validación
        plt.plot(self.descaled_test.index, self.descaled_test[target_column], 
                label = f"Validation ({target_column})", color = "orange", alpha = 0.7)

        # Agregar predicciones de entrenamiento
        if train_preds is not None:
            start_idx_train = 120  # Índice inicial de las predicciones de entrenamiento
            train_pred_indices = range(start_idx_train, start_idx_train + len(train_preds))  # Índices
            plt.plot(train_pred_indices, train_preds, 
                    label = "Train Predictions", color = "green", linestyle = "--", alpha = 0.7)

        # Agregar predicciones de validación
        if test_preds is not None:
            # Índice inicial de las predicciones de validación (fin de train + 121)
            start_idx_test = self.descaled_train.index[-1] + 121
            test_pred_indices = range(start_idx_test, start_idx_test + len(test_preds))  # Índices
            plt.plot(test_pred_indices, test_preds, 
                    label = "Validation Predictions", color = "red", linestyle="--", alpha = 0.7)

        # Configuración del gráfico
        plt.title(f"Comportamiento del Bitcoin ({target_column}) y Predicciones")
        plt.xlabel("Índice temporal")
        plt.ylabel(f"Valor de {target_column}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Mostrar gráfico
        plt.show()
