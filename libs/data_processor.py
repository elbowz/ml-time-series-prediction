import numpy as np
import pandas as pd
from math import ceil
from typing import Union, Optional, Tuple, Iterable


class DataLSTMProcessor:
    def __init__(self, dataframe: Union[str, pd.DataFrame], split: int, columns: list, len_train_window: int, normalise: bool):
        """
        A class for loading and transforming data for the LSTM model

        :param dataframe: filename or pandas DataFrame
        :param split: [0-1] TrainingSet length
        :param columns: list of columns to keep
        :param len_train_window: train window length
        :param normalise: normalise data
        """
        if type(dataframe) is str:
            dataframe = pd.read_csv(dataframe)

        # Scaler MinMax
        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # dataframe = pd.DataFrame(scaler.fit_transform(self.data), columns=cols)

        i_split = int(len(dataframe) * split)
        self.data = dataframe.get(columns)
        self.data_train = dataframe.get(columns).values[:i_split]
        self.data_test = dataframe.get(columns).values[i_split:]

        self.data_train_index = dataframe.index.values[:i_split]
        self.data_test_index = dataframe.index.values[i_split:]

        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)

        self.len_train_window = len_train_window
        self.normalise = normalise
        self.test_x = None
        self.test_y = None

    def get_test_data(self) -> Tuple[np.array, np.array]:
        """
        Create input (x) and output (y) TestSet window

        :return: x and y test
        """
        data_windows = []

        # create the sliding windows step by step
        for i in range(self.len_test - self.len_train_window + 1):
            data_windows.append(self.data_test[i:i+self.len_train_window])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if self.normalise else data_windows

        # keep all windows without last element (what we want predict)
        self.test_x = data_windows[:, :-1]
        # keep only the last element of all windows, precisely the first column (ie. close)
        self.test_y = data_windows[:, -1, [0]]

        return self.test_x, self.test_y

    def get_train_data(self) -> Tuple[np.array, np.array]:
        """
        Create input (x) and output (y) TrainingSet window.
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        """
        data_x = []
        data_y = []

        for i in range(self.len_train - self.len_train_window + 1):
            x, y = self._next_window(i)
            data_x.append(x)
            data_y.append(y)

        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, batch_size: int) -> Iterable[Tuple[np.array, np.array]]:
        """
        Yield a generator of TrainingSet.
        Useful with small amount of ram or huge TrainingSet

        :param batch_size: batch_size length
        """
        i = 0
        while i < (self.len_train - self.len_train_window + 1):
            x_batch = []
            y_batch = []

            for b in range(batch_size):
                if i >= (self.len_train - self.len_train_window + 1):

                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0

                x, y = self._next_window(i)

                x_batch.append(x)
                y_batch.append(y)
                i += 1

            yield np.array(x_batch), np.array(y_batch)

    def get_steps_per_epoch(self, batch_size: int) -> int:
        """
        Retrieve the steps (batch of samples) per epoch. Needed with keras *train_generator* method

        :param batch_size: batch_size
        """
        return ceil((self.len_train - self.len_train_window + 1) / batch_size)

    def _next_window(self, i: int) -> np.array:
        """
        Generates the next data window from the given index location i

        :param i: first element of window
        """
        window = self.data_train[i:i+self.len_train_window]
        window = self.normalise_windows(window, single_window=True)[0] if self.normalise else window
        x = window[:-1]

        # keep last element in the list (window)
        # and the first column (ie. close)
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data: pd.DataFrame, single_window: bool = False):
        """
        Normalise window with a base value of first element window

        :param window_data: window(s) to normalize
        :param single_window: window_data is a single window
        """
        normalised_data = []
        window_data = [window_data] if single_window else window_data

        for window in window_data:
            normalised_window = []

            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)

            # reshape and transpose array back into original multidimensional format
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)

        return np.array(normalised_data)

    def inverse_normalise(self, normalised_data: Optional[np.array] = None, original_data: Optional[np.array] = None) -> np.array:
        """
        Inverse normalisation

        :param normalised_data: data to invert
        :param original_data: data used to normalize. Should be normalised_data + len_train_window length
        """
        de_normalised_data = []

        for index, normalised_value in enumerate(normalised_data):
            de_normalised_value = (original_data[index]) * (normalised_value + 1)
            de_normalised_data.append(de_normalised_value)

        return np.array(de_normalised_data)

    def inverse_normalise_y(self, normalised_data: np.array = None) -> np.array:
        """
        Inverse normalisation for output y. Use TestSet input (y) as original_data.

        :param normalised_data: data to invert
        """
        if normalised_data is None:
            normalised_data = self.test_y.reshape(-1)

        original_data = self.data_test[:, 0]

        return self.inverse_normalise(normalised_data, original_data)
