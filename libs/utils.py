import json
import logging
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple, Iterable

# Setup plot
plt.rcParams.update({'figure.figsize': (12, 6), 'font.size': 8})

# Setup logging.
logger = logging.getLogger('main-log')


class Timer:
    def __init__(self):
        """Report gap time from *start* to *stop*"""
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self) -> dt:
        end_dt = dt.datetime.now()
        return end_dt - self.start_dt


def load_config(config_path: str) -> dict:
    """ Open and decode config json """
    config = False

    try:
        config_file = open(config_path, 'r')
        config = json.load(config_file)

    except IOError:
        logger.error('No Config file found: {}'.format(args.config_filename))
        exit(1)

    except ValueError:
        logger.error('Config file is not a correct JSON format: {}'.format(args.config_filename))
        exit(1)

    return config


def df_print(df, max_rows=None, my_print=print, head=''):
    """Print pandas dataframe with some configuration"""
    with pd.option_context('display.max_rows', max_rows):
        my_print('{}\n{}'.format(head, df))


def plot_line_graph(x, ylist, fmt=('bo-', 'g^-', 'rv-', 'c>:', 'm<-'),
                    labels=None, title='', xlabel='', ylabel='', ticks_count=None,
                    linewidth=1, markersize=12, figsize=(12, 6)):
    """Plot multiple line on single graph"""
    plt.figure(figsize=figsize)

    for index, y in enumerate(ylist):
        label = labels[index] if labels is not None else None
        plt.plot(x, y, fmt[index], label=label, linewidth=linewidth, markersize=markersize)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if ticks_count is not None:
        tick_step = int(len(x) / ticks_count)
        plt.xticks(x[::tick_step], rotation=45)

    plt.legend(loc='best')


def plot_results_multiple(y_predicted, y_true, prediction_len, title='', xlabel='', ylabel='', figsize=(12, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    ax.plot(y_true, label='Stock value')

    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(y_predicted):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='LSTM Prediction')
        plt.legend()
