#!/usr/bin/env python3.6

# inspired to https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction

"""
# How to install dependencies on Arch Linux
# note: we use python 3.6 beacuse tensorflow don't support python3.7 
pacaur -S python36
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3.6 get-pip.py
sudo pip3.6 install -U keras
sudo pip3.6 install -U tensorflow
# https://github.com/RomelTorres/alpha_vantage
sudo pip3.6 install -U alpha_vantage
pacaur -S ta-lib
# https://mrjbq7.github.io/ta-lib/
sudo pip3.6 install -U TA-lib
sudo pip3.6 install -U requests
sudo pip3.6 install -U scikit-learn
sudo pip3.6 install -U matplotlib
sudo pip3.6 install -U pandas
sudo pacman -S tk
"""

# system libs
import os
import sys
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# internal libs
from libs.stock_data import StockData
from libs.stock_strategies import StockStrategies
from libs.data_processor import DataLSTMProcessor
from libs.model import LSTMModel, StrategyModel
from libs.utils import Timer, plot_results_multiple, plot_line_graph, df_print, load_config

# Setup logging.
logger = logging.getLogger('main-log')
logger.setLevel(logging.DEBUG)

str_out = logging.StreamHandler(sys.stdout)
str_out.setLevel(logging.DEBUG)
str_out.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%I:%M:%S %p'))
logger.addHandler(str_out)

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# global vars
__default_data_dir = './data_set'
__default_models_dir = './saved_models'


def main():
    # Setup script argument parsing
    args = setup_arg_parser()

    # open and decode config json
    config = load_config(args.config_filename)
    logger.debug(f'Config json file ({args.config_filename}) loaded')

    # manage stock data
    symbol_data = StockData(config, __default_data_dir)

    # if data csv is not yet saved or user force to overwrite
    if not os.path.isfile(symbol_data.symbol_csv_path) or args.symbol_csv_overwrite:

        # load stock data from alpha vantage
        symbol_dataframe, symbol_meta_data = symbol_data.load()
        logger.debug(f"Retrived info: {symbol_meta_data['Information']} | "
                     f"Updated: {symbol_meta_data['Last Refreshed']} | "
                     f"Time zone: {symbol_meta_data['Time Zone']}")

        # save on csv
        symbol_data.save()
        logger.info(f"Saved symbol ({config['data']['symbol']}) data in {symbol_data.symbol_csv_path}")

    # load from csv
    symbol_dataframe = symbol_data.load_from_csv()
    logger.debug(f'Symbol data file ({symbol_data.symbol_csv_path}) loaded')
    df_print(symbol_dataframe, 8, logger.debug, 'Extract of first/last 4 rows:')

    # calculate (through talib) and add indicators to symbol_dataframe
    symbol_data.add_indicators()

    # rows slicing, drop NaN value, etc..
    symbol_dataframe = symbol_data.data_preprocessing()
    df_print(symbol_dataframe, 8, logger.debug, 'Extract of first/last 4 rows:')

    # is a market stock
    if config['data']['symbol'] != 'function':
        # symbol chart with indicators (and volume)
        symbol_data.plot_symbol(f"{config['data']['symbol']} - Stock DataSet",
                                list(config['data'].get('indicators').keys()) + ['volume'])

    train_test_split = config['data']['train_test_split']
    logger.info(f'Splitting DataSet (TrainingSet / TestSet): {train_test_split} / {(1 - train_test_split):.2f}')
    logger.info('Training window{} normalized length: {}'
                .format('' if config['data']['normalise'] else ' NOT', config['data']['len_train_window']))

    # prepare DataSet (split, shape, normalisation, etc...)
    symbol_lstm_data = DataLSTMProcessor(
        dataframe=symbol_dataframe,
        split=config['data']['train_test_split'],
        columns=config['data']['columns'],
        len_train_window=config['data']['len_train_window'],
        normalise=config['data']['normalise']
    )

    # get TestSet
    x_test, y_test = symbol_lstm_data.get_test_data()

    # create LSTM model
    lstm_model = LSTMModel(config, __default_models_dir)

    # check if there is a saved model that "match" with config
    saved_model_path = lstm_model.get_saved_model_path()

    if os.path.isfile(saved_model_path) and args.force_traing is False:
        # load the model (avoid re-training)

        logger.info(f'Loading model from file: {saved_model_path}')
        lstm_model.load_model(saved_model_path)
    else:
        # train the model
        train_lstm_model(lstm_model, symbol_lstm_data, config, args.in_memory_training, x_test, y_test)

    logger.info('Forecasting...')
    len_train_window = config['data']['len_train_window']

    # calculate different kind of predictions
    y_predictions = lstm_model.predict_point_by_point(x_test)
    y_predictions_full = lstm_model.predict_sequence_full(x_test, len_train_window)
    y_predictions_multiple = lstm_model.predict_sequences_multiple(x_test, len_train_window)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    logger.info(f'Prediction metrics: {mean_squared_error(y_test, y_predictions)} Mean squared error, '
                f'{mean_absolute_error(y_test, y_predictions)} Mean absolute error')

    # is a market stock
    if config['data']['symbol'] == 'function':
        # plot y_test and y_prediction as normalized data
        plot_line_graph(range(len(y_predictions)),
                        [y_test.reshape(-1)] + [y_predictions],
                        fmt=('b-', 'y-'),
                        labels=['True Data', 'LSTM Model Prediction'],
                        title=f"{config['data']['filename']} - Function TestSet prediction by LSTM Model",
                        ylabel='f(x)')

        # plot y_predictions_multiple
        plot_results_multiple(y_predictions_multiple, y_test, len_train_window,
                              f"{config['data']['filename']} - Function TestSet MULTIPLE prediction by LSTM Model",
                              ylabel='f(x)')

        # plot y_predictions_full
        plot_line_graph(range(len(y_predictions_full)),
                        [y_test.reshape(-1)] + [y_predictions_full],
                        fmt=('b-', 'y-'),
                        labels=['True Data', 'LSTM Model Prediction'],
                        title=f"{config['data']['filename']} - Function TestSet FULL prediction by LSTM Model",
                        ylabel='f(x)')

        return 0

    # plot y_test and y_prediction as normalized data
    plot_line_graph(symbol_lstm_data.data_test_index[len_train_window-1:],
                    [y_test.reshape(-1)] + [y_predictions],
                    fmt=('b-', 'y-'),
                    labels=['True Data', 'LSTM Model Prediction'],
                    title=f"{config['data']['symbol']} - Stock TestSet normalized prediction by LSTM Model",
                    xlabel='Data',
                    ylabel='Stock value')

    # plot y_predictions_multiple
    plot_results_multiple(y_predictions_multiple, y_test, len_train_window,
                          f"{config['data']['symbol']} - Stock TestSet normalized MULTIPLE prediction by LSTM Model",
                          xlabel='Data',
                          ylabel='Stock value')

    # plot y_predictions_full
    plot_line_graph(symbol_lstm_data.data_test_index[len_train_window-1:],
                    [y_test.reshape(-1)] + [y_predictions_full],
                    fmt=('b-', 'y-'),
                    labels=['True Data', 'LSTM Model Prediction'],
                    title=f"{config['data']['symbol']} - Stock TestSet normalized FULL prediction by LSTM Model",
                    xlabel='Data',
                    ylabel='Stock value')


    # create strategies DataFrame by random choice, LSTM prediction and all indicators
    symbol_strategies = StockStrategies(config, y_test, y_predictions, symbol_lstm_data, symbol_dataframe)

    strategies_dataframe = symbol_strategies.strategies_dataframe
    symbol_predictions_dataframe = symbol_strategies.symbol_predictions_dataframe

    df_print(strategies_dataframe, 8, logger.debug, 'Extract of first/last 4 rows of Strategies DataSet:')

    """ Print Hit rate (accuracy) and Gain about each indicators """
    y_true = strategies_dataframe['optimum_signal'].values
    symbol_start_value = strategies_dataframe['y_denormalized'][0]

    logger.info('Strategy evaluation')
    logger.info('Strategy       | Total |  Hit  | Hit Perc. | Gain unit | Gain perc.')

    for indicator_name in StockStrategies.strategies_name:
        signal_label = indicator_name + '_signal'
        gain = symbol_strategies.strategy_gain(signal_label)

        # remove first rows where indicators have not set the strategy (eg. for RSI indicators)
        strategies_dataframe_trimmed = strategies_dataframe.dropna(subset=[signal_label])
        len_signal = strategies_dataframe_trimmed.shape[0]

        if len_signal:
            hit = accuracy_score(y_true[-len_signal:], strategies_dataframe_trimmed[signal_label].values, normalize=False)

            logger.info(f'{indicator_name.capitalize():<14} | {len_signal:^5} | {hit:^5} | {(hit / len_signal):^9.2%} | '
                        f'{gain:^9.2f} | {(1 / symbol_start_value * gain):>10.2%}')

    logger.info(f'note: start investment of {symbol_start_value:.2f} $ (symbol unit value)')

    # plot a graph with entry (and exit point) given by strategy lstm_signal
    symbol_data.plot_symbol(f"{config['data']['symbol']} - Stock TestSet strategy by LSTM Model",
                            ['rsi', 'volume'],
                            symbol_predictions_dataframe,
                            strategies_dataframe,
                            'lstm_signal')

    """ Create classifier model to find the best strategy """
    strategy_model = StrategyModel(config, strategies_dataframe)

    df_print(strategy_model.dataset_dataframe, 8, logger.debug, 'Extract of first/last 4 rows of Strategies DataSet for model encoded:')

    logger.info('Model strategy evaluation')

    y_mlp_predictions = strategy_model.mlp_predict()
    hit = accuracy_score(strategy_model.y_test, y_mlp_predictions, normalize=False)

    logger.info(f'{"Mlp":<14} | {len(y_mlp_predictions):^5} | {hit:^5} | '
                f'{(hit / len(y_mlp_predictions)):^9.2%}')

    y_random_forest_predictions = strategy_model.random_forest_predict()
    hit = accuracy_score(strategy_model.y_test, y_mlp_predictions, normalize=False)

    logger.info(f'{"Random forest":<14} | {len(y_random_forest_predictions):^5} | '
                f'{hit:^5} | {(hit / len(y_random_forest_predictions)):^9.2%}')

    logger.info(f"note: we cannot compare with above strategies because is a subset "
                f"(TestSet split: {config['strategy_model']['train_test_split']})")

    return 0


def setup_arg_parser():
    # Script argument parsing
    parser = argparse.ArgumentParser(description='Time series prediction and strategies on stock markets - Machine learning a.a. 2018/19',
                                     epilog=' coded by: Emanuele Palombo')

    parser.add_argument('config_filename', metavar='CFG_FILENAME', type=str, help='string (eg. msft.json) - Config filename')

    parser.add_argument('--in-memory-training', '-m', dest='in_memory_training', action='store_true', default=False,
                        help='(default False) - training will be done whole in memory. Take care to have RAM!')

    parser.add_argument('--symbol-data-overwrite', '-d', dest='symbol_csv_overwrite', action='store_true', default=False,
                        help=f'(default False) - get new data for symbol and overwrite the releated csv file stored in {__default_data_dir}')

    parser.add_argument('--force-train', '-f', dest='force_traing', action='store_true', default=False,
                        help=f'ignore config file change or saved model and force to retrain ')

    return parser.parse_args()


def train_lstm_model(lstm_model, symbol_lstm_data, config, in_memory_training, x_test, y_test):
    timer = Timer()
    timer.start()

    # build the model (from config)
    lstm_model.build_model()

    history = None

    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']

    # training the model
    if in_memory_training:
        # in-memory training
        logger.info(f'Training (in-memory): {epochs} epochs, {batch_size} batch size')

        x, y = data.get_train_data()
        lstm_model.train(x, y, epochs, batch_size)
    else:
        # out-of memory generative training
        steps_per_epoch = symbol_lstm_data.get_steps_per_epoch(batch_size)
        data_gen = symbol_lstm_data.generate_train_batch(batch_size)

        logger.info(f'Training (out-of memory): {epochs} epochs, {batch_size} batch size, {steps_per_epoch} batches per epoch')

        history = lstm_model.train_generator(
            data_gen=data_gen,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=(x_test, y_test)
        )

    logger.info(f'Model compiled and trained in: {timer.stop()}')
    logger.info(f'Model saved as: {lstm_model.get_saved_model_path()}')
    # logger.debug('Metrics by batch:\n{}'.format(history['batchs_history']))
    logger.debug('Metrics by epochs:\n{}'.format(history['epochs_history']))

    # plot loss value during training
    plot_line_graph(range(len(history['batchs_history']['loss'])),
                    [history['batchs_history']['loss']] + [history['batchs_history']['mean_absolute_error']],
                    labels=[f"Loss ({config['model']['loss']})", 'Mean absolute error'],
                    title=f"{config['data']['symbol']} - Loss value during training ({config['model']['loss']})",
                    xlabel='Batch',
                    ylabel=f"Loss ({config['model']['loss']})",
                    linewidth=1.6, markersize=4)


if __name__ == '__main__':

    main()

    plt.show()

    exit(0)
