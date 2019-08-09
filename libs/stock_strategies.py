import logging
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Iterable

# internal libs
from libs.data_processor import DataLSTMProcessor

# Setup logging.
logger = logging.getLogger('main-log')


class StockStrategies:
    strategies_name = ('optimum', 'random', 'lstm', 'rsi', 'bbands', 'sma')
    buy = 'buy'
    sell = 'sell'

    def __init__(self, config: dict,  y_true: np.array, y_predictions: np.array, symbol_lstm_data: DataLSTMProcessor, symbol_dataframe: pd.DataFrame):
        """
        A class for create a strategies dataframe based on real data (y_true), lstm (y_predictions), random choice and various indicators
        """
        self.config = config

        self.strategies_dataframe = None
        self.symbol_predictions_dataframe = None

        self._create(y_true, y_predictions, symbol_lstm_data, symbol_dataframe)

    def _create(self, y_true: np.array, y_predictions: np.array, symbol_lstm_data: DataLSTMProcessor, symbol_dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """ Crete a DataFrame with buy and sell strategies based on predicionts and indicators """
        y_true_normalized = y_true.reshape(-1)
        y_true_de_normalised = symbol_lstm_data.inverse_normalise_y()
        lstm_predictions_denormalized = symbol_lstm_data.inverse_normalise_y(y_predictions)

        len_train_window = self.config['data']['len_train_window']

        # create DataFrame with trading strategies (re-apply date as index)
        self.strategies_dataframe = pd.DataFrame(index=symbol_lstm_data.data_test_index[len_train_window-1:-1], data={
            'y': y_true_normalized[:-1],
            'y_plus_one': y_true_normalized[1:],
            'lstm_predictions': y_predictions[:-1],
            'lstm_predictions_plus_one': y_predictions[1:],
            'y_denormalized': y_true_de_normalised[:-1],
            'y_denormalized_plus_one': y_true_de_normalised[1:],
            'lstm_predictions_denormalized': lstm_predictions_denormalized[:-1],
            'lstm_predictions_denormalized_plus_one': lstm_predictions_denormalized[1:],
        })

        # keep "in sync" symbol_predictions_dataframe with symbol_dataframe selecting/slicing row over the common index (date)
        # this allow to make strategies on indicators (calculated in the StockData)
        self.symbol_predictions_dataframe = symbol_dataframe.loc[self.strategies_dataframe.index]

        # add whole strategies to strategies_dataframe
        for strategy in StockStrategies.strategies_name:
            method_to_call = getattr(self, 'add_strategy_' + strategy)
            method_to_call()

        # if NaN, copy value from above row (same column)
        self.strategies_dataframe.ffill(inplace=True)

    def add_strategy_optimum(self):
        strategies_dataframe = self.strategies_dataframe

        # prefect trading strategy: based TestSet y true data (we know the future)
        strategies_dataframe.loc[(strategies_dataframe['y'] <= strategies_dataframe['y_plus_one']), 'optimum_signal'] = StockStrategies.buy
        strategies_dataframe.loc[(strategies_dataframe['y'] > strategies_dataframe['y_plus_one']), 'optimum_signal'] = StockStrategies.sell

    def add_strategy_random(self):
        strategies_dataframe = self.strategies_dataframe

        # random strategy: this is what you think
        strategies_dataframe['random_signal'] = np.random.choice([StockStrategies.buy, StockStrategies.sell], strategies_dataframe.shape[0])

    def add_strategy_lstm(self):
        strategies_dataframe = self.strategies_dataframe

        # lstm trading strategy: signal based on lstm prediction over symbol value
        # NOTE: strategies are not made on denormalized data because the process of denormalization lose informations
        strategies_dataframe.loc[(strategies_dataframe['y'] <= strategies_dataframe['lstm_predictions_plus_one']), 'lstm_signal'] = StockStrategies.buy
        strategies_dataframe.loc[(strategies_dataframe['y'] > strategies_dataframe['lstm_predictions_plus_one']), 'lstm_signal'] = StockStrategies.sell

    def add_strategy_rsi(self):
        strategies_dataframe = self.strategies_dataframe
        symbol_predictions_dataframe = self.symbol_predictions_dataframe

        if 'rsi' in symbol_predictions_dataframe.columns:
            # rsi trading strategies: triggered when rsi cross an up and down thresholds
            strategies_dataframe.loc[(symbol_predictions_dataframe['rsi'] < 30), 'rsi_signal'] = StockStrategies.buy
            strategies_dataframe.loc[(symbol_predictions_dataframe['rsi'] > 70), 'rsi_signal'] = StockStrategies.sell

    def add_strategy_bbands(self):
        strategies_dataframe = self.strategies_dataframe
        symbol_predictions_dataframe = self.symbol_predictions_dataframe

        if 'bbands' in symbol_predictions_dataframe.columns:
            # bollinger bands trading strategies: triggered when symbol value cross the bollinger bands
            strategies_dataframe.loc[(symbol_predictions_dataframe['bbands'] < 0), 'bbands_signal'] = StockStrategies.buy
            strategies_dataframe.loc[(symbol_predictions_dataframe['bbands'] > 1), 'bbands_signal'] = StockStrategies.sell

    def add_strategy_sma(self):
        strategies_dataframe = self.strategies_dataframe
        symbol_predictions_dataframe = self.symbol_predictions_dataframe

        if 'sma' in symbol_predictions_dataframe.columns:
            # sma trading strategies: triggered when symbol value cross the sma
            strategies_dataframe.loc[(symbol_predictions_dataframe['sma'] < symbol_predictions_dataframe['close']), 'sma_signal'] = StockStrategies.buy
            strategies_dataframe.loc[(symbol_predictions_dataframe['sma'] > symbol_predictions_dataframe['close']), 'sma_signal'] = StockStrategies.sell

    def strategy_gain(self, column_signal: str, y_denormalized:  Optional[str] = 'y_denormalized',
                      y_denormalized_plus_one: Optional[str] = 'y_denormalized_plus_one') -> Tuple[float, float]:
        """Calculate gain following the predicted signal"""
        strategies_dataframe = self.strategies_dataframe

        tot_gain = 0
        for index, row in strategies_dataframe.iterrows():
            # calculate gain (positive or negative contribute) following the predicted signal
            gain = row[y_denormalized_plus_one] - row[y_denormalized]

            if row[column_signal] == StockStrategies.buy:
                tot_gain += gain
            elif row[column_signal] == StockStrategies.sell:
                tot_gain -= gain

        return tot_gain
