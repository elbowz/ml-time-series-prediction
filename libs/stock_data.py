import os
import talib
import logging
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from typing import Union, Optional, Tuple, Iterable

# Setup logging.
logger = logging.getLogger('main-log')

# Setup plot
plt.rcParams.update({'figure.figsize': (12, 6), 'font.size': 8})


class StockData:
    alpha_vantage_key = 'GZFL51FOA0J0J0JW'
    value_column_name = 'close'
    data_columns_name = ['open', 'high', 'low', 'close-bad', 'close', 'volume', 'dividend', 'split coefficient']

    def __init__(self, config: dict, data_dir: str):
        """
        A class for loading (alpha vantage), save (csv) and process stock market data

        :param dataframe: config dict
        :param data_dir: path to directory where find csv
        """
        self.config = config
        self.config_interval = self.config['data'].get('interval', '1day')
        self.symbol_csv_path = os.path.join(data_dir, config['data']['filename'])

        self.symbol_dataframe = None
        self.strategies_dataframe = None

    def load(self) -> Tuple[pd.DataFrame, dict]:
        """ Retrive from alpha vantage and save symbol values in csv """
        av_ts = TimeSeries(key=StockData.alpha_vantage_key, output_format='pandas')

        # TODO: add weekly and monthly (could be usefull?)
        if self.config_interval in ('1min', '5min', '15min', '30min', '60min'):
            # get intraday values
            symbol_dataframe, symbol_meta_data = av_ts.get_intraday(symbol=self.config['data']['symbol'],
                                                               interval=self.config['data']['interval'],
                                                               outputsize='full')
        else:
            # get daily values
            symbol_dataframe, symbol_meta_data = av_ts.get_daily_adjusted(symbol=self.config['data']['symbol'],
                                                                     outputsize='full')

        # rename alpha_vantage default columns name
        symbol_dataframe.columns = StockData.data_columns_name
        self.symbol_dataframe = symbol_dataframe

        # strip first 3 char from symbol_meta_data key (eg. '1. Information' => 'Information')
        symbol_meta_data = {key[3:]: value for (key, value) in symbol_meta_data.items()}

        self.symbol_dataframe = symbol_dataframe

        return self.symbol_dataframe, symbol_meta_data

    def load_from_csv(self) -> pd.DataFrame:
        """ Retrive from csv file """
        # define datetime format
        if self.config_interval in ('1min', '5min', '15min', '30min', '60min'):
            dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        else:
            dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

        # choose column (or autoincrement) for index (ie. x axis on plot)
        index_col = 0
        parse_dates = ['date']

        if self.config['data'].get('index_col') != 'date':
            index_col = None
            parse_dates = False

        # retrive pandas dataframe from saved csv
        self.symbol_dataframe = pd.read_csv(self.symbol_csv_path, parse_dates=parse_dates,
                                            date_parser=dateparse, index_col=index_col)

        return self.symbol_dataframe

    def save(self, symbol_csv_path: Optional[str] = None, symbol_dataframe: Optional[pd.DataFrame] = None) -> Optional[str]:
        """ Save symbol_dataframe to csv file """
        if symbol_csv_path is None:
            symbol_csv_path = self.symbol_csv_path

        if symbol_dataframe is None:
            symbol_dataframe = self.symbol_dataframe

        # write csv file
        return symbol_dataframe.to_csv(symbol_csv_path, encoding='utf-8')

    def add_indicators(self, symbol_dataframe: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """ Add technical analysis indicators  to symbol_dataframe"""
        if symbol_dataframe is None:
            symbol_dataframe = self.symbol_dataframe

        indicators = self.config['data'].get('indicators', {})

        # get stock value
        if len(indicators):
            close = symbol_dataframe[StockData.value_column_name].values

        # calculate (through talib) and add indicators to symbol_dataframe
        for indicator, opts in indicators.items():

            logger.debug('Adding indicator to data: {:<6} ( {} )'.format(indicator, str(opts)[1:-1].replace("'", '')))

            if indicator == 'bbands':
                up, mid, low = talib.BBANDS(close, **opts)

                # create a "new indicators": >1 => sell | <0 buy (see bollinger bands definition)
                bbands = (close - low) / (up - low)

                symbol_dataframe = symbol_dataframe.assign(bb_up=pd.Series(up).values)
                symbol_dataframe = symbol_dataframe.assign(bb_down=pd.Series(low).values)
                symbol_dataframe = symbol_dataframe.assign(bbands=pd.Series(bbands).values)

            elif indicator == 'sma':
                sma = talib.SMA(close, **opts)

                symbol_dataframe = symbol_dataframe.assign(sma=pd.Series(sma).values)

            elif indicator == 'rsi':
                rsi = talib.RSI(close, **opts)

                symbol_dataframe = symbol_dataframe.assign(rsi=pd.Series(rsi).values)

        self.symbol_dataframe = symbol_dataframe

        return self.symbol_dataframe

    def data_preprocessing(self, symbol_dataframe: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """ Rows slicing, drop NaN value, etc... """
        if symbol_dataframe is None:
            symbol_dataframe = self.symbol_dataframe

        start_from = self.config['data'].get('start_from')
        end_to = self.config['data'].get('end_to')
        symbol_dataframe = symbol_dataframe.loc[start_from:end_to]

        logger.debug(f'Sliced data from {start_from} to {end_to}')

        # drop row if a NaN is present
        len_nan_row = symbol_dataframe.shape[0]
        symbol_dataframe.dropna(inplace=True)
        len_nan_row -= symbol_dataframe.shape[0]

        logger.debug(f'Removed NaN row (where indicators are still 0 due to timeperiod opt): {len_nan_row} rows')

        self.symbol_dataframe = symbol_dataframe

        return self.symbol_dataframe

    def plot_symbol(self, title='',
                    indicators: Optional[list] = ('bbands', 'rsi', 'sma', 'volume'),
                    symbol_dataframe: Optional[pd.DataFrame] = None,
                    strategy_dataframe: Optional[pd.DataFrame] = None,
                    strategy_column: Optional[str] = None):
        """ Plot a custom symbol chart with indicators and strategies """
        if symbol_dataframe is None:
            symbol_dataframe = self.symbol_dataframe

        plt.figure()

        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=4, colspan=1)

        ax1.set_title(title)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock value')

        if 'rsi' in indicators:
            ax2 = plt.subplot2grid((6, 1), (4, 0), rowspan=1, colspan=1, sharex=ax1)
            ax2.set_ylabel('Rsi')
            ax2.set_xlabel('Date')
        if 'volume' in indicators:
            ax3 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
            ax3.set_ylabel('Volume')
            ax3.set_xlabel('Date')

        ax1.plot(symbol_dataframe.index, symbol_dataframe['close'], label='Stock value', linewidth=1.2)

        if 'bbands' in indicators:
            ax1.plot(symbol_dataframe.index, symbol_dataframe['bb_up'], label='BBand Up', linewidth=0.6)
            ax1.plot(symbol_dataframe.index, symbol_dataframe['bb_down'], label='BBand Down', linewidth=0.6)
            ax1.fill_between(symbol_dataframe.index, y1=symbol_dataframe['bb_down'], y2=symbol_dataframe['bb_up'], color='#adccff', alpha='0.3')

        if 'sma' in indicators:
            ax1.plot(symbol_dataframe.index, symbol_dataframe['sma'], label='Sma', linewidth=1.0)

        if 'rsi' in indicators:
            ax2.plot(symbol_dataframe.index, symbol_dataframe['rsi'])
            ax2.fill_between(symbol_dataframe.index, y1=30, y2=70, color='#adccff', alpha='0.3')

        if 'volume' in indicators:
            ax3.bar(symbol_dataframe.index, symbol_dataframe['volume'])

        if strategy_dataframe is not None and strategy_column is not None:
            # add prediction value
            # ax1.plot(symbol_dataframe.index, strategy_dataframe['lstm_predictions_denormalized'], label='LSTM Prediction', linewidth=1.2)

            last_signal = None
            # add orders enter points (buy/sell) following strategy
            for index_x, row in strategy_dataframe.iterrows():
                signal = row[strategy_column]

                if signal == 'buy' and signal != last_signal:
                    ax1.scatter(x=index_x, y=strategy_dataframe.loc[index_x, 'y_denormalized'], marker='^', color='green', s=20)
                elif signal == 'sell' and signal != last_signal:
                    ax1.scatter(x=index_x, y=strategy_dataframe.loc[index_x, 'y_denormalized'], marker='v', color='red', s=20)

                last_signal = signal

        ax1.legend(loc='best')
