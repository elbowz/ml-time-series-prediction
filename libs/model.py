import os
import json
import hashlib
import numpy as np
import pandas as pd
from numpy import newaxis
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback as Keras_callabacks
from typing import Union, Optional, Tuple, Iterable

# internal libs
from libs.stock_strategies import StockStrategies


class BatchHistory(Keras_callabacks):
    @property
    def history(self):
        return dict(self._history)

    def on_train_begin(self, logs={}):
        self._history = defaultdict(list)

    def on_batch_end(self, batch, logs={}):
        for key in logs.keys():
            self._history[key].append(logs.get(key))


class LSTMModel:
    def __init__(self, config: dict, save_model_dir: str = './'):
        """A class for an building an LSTM model"""
        self.model = Sequential()
        self.config = config
        self.save_model_dir = save_model_dir

    def get_saved_model_path(self) -> str:
        """Get the whole module_path made by: *save_model_dir/save_filename-MD5(config).mdl*"""
        config_json = json.dumps(self.config).encode('utf-8')
        config_hash = hashlib.md5(config_json).hexdigest()

        saved_module_path = os.path.join(self.save_model_dir, self.config['model']['save_filename'])
        saved_module_path = '{}-{}.mdl'.format(saved_module_path, config_hash)

        return saved_module_path

    def load_model(self, module_path: Optional[str] = None):
        """Load saved model from *module_path* to avoid re-training"""
        if module_path is None:
            module_path = self.get_saved_model_path()

        self.model = load_model(module_path)

    def build_model(self):
        """Build/compile model starting from config"""
        for layer in self.config['model']['layers']:

            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=self.config['model']['loss'],
                           optimizer=self.config['model']['optimizer'],
                           metrics=['mse', 'mae', 'mape', 'cosine'])

    def train(self, x: np.array, y: np.array, epochs: int, batch_size: Optional[int] = None) -> dict:
        """Classic training, see *train_generator* with generator. More information of params to: https://keras.io/models/sequential/#fit"""
        saved_model_path = self.get_saved_model_path()

        batch_history = BatchHistory()

        callbacks = [
            batch_history,
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=saved_model_path, monitor='loss', save_best_only=True)
        ]

        history = self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        self.model.save(saved_model_path)

        return {'epochs_history': history.history, 'batchs_history': batch_history.history}

    def train_generator(self, data_gen: Iterable[Tuple[np.array, np.array]],
                        epochs: int, steps_per_epoch: int,
                        validation_data: Tuple[np.array, np.array]) -> dict:
        """
        Training through a iterable (generator) to avoid to fill ram. More information of params to:
        https://keras.io/models/sequential/#fit_generator
        """
        saved_model_path = self.get_saved_model_path()

        batch_history = BatchHistory()

        callbacks = [
            batch_history,
            ModelCheckpoint(filepath=saved_model_path, monitor='loss', save_best_only=True)
        ]
        history = self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1,
            validation_data=validation_data
        )

        return {'epochs_history': history.history, 'batchs_history': batch_history.history}

    def predict_point_by_point(self, data: np.array) -> np.array:
        """
        Predict each timestep, only predicting 1 step ahead each time.
        Use always real TestSet points for make a predictions.
        """
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))

        return predicted

    def predict_sequence_full(self, data: np.array, window_size: int) -> list:
        """
        Shift the window by 1 new prediction each time, re-run predictions on new window.
        Only the first *window_size* points are real, come from the TestSet (seed), after
        that model make prediction only on prediction.
        """

        predicted = []
        curr_frame = data[0]

        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            # shift 1 step
            curr_frame = curr_frame[1:]
            # append the predicted value
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)

        return predicted

    def predict_sequences_multiple(self, data: np.array, window_size: int, prediction_len: Optional[int] = None) -> list:
        """
        Predict sequence of *prediction_len* steps before shifting prediction run forward by *prediction_len* steps.
        It's a blend of *predict_point_by_point* and *predict_sequence_full* methods, this have the same
        behaviour of *predict_sequence_full* for the first *prediction_len*, after that shift the window of a
        full *prediction_len* and restart on real TestSet points.

        default: *prediction_len* = *window_size*
        """
        if prediction_len is None:
            prediction_len = window_size

        prediction_seqs = []

        for i in range(int(len(data) / prediction_len)):
            # run forward by prediction_len steps
            curr_frame = data[i * prediction_len]
            predicted = []

            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                # shift 1 step
                curr_frame = curr_frame[1:]
                # append the predicted value
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)

            prediction_seqs.append(predicted)

        return prediction_seqs


class StrategyModel:
    y_column_name = 'optimum_signal'

    def __init__(self, config: dict, strategies_dataframe: pd.DataFrame):
        """ A class for an building an classifier model for choose the best strategy """
        self.config = config['strategy_model']
        self.dataset_dataframe = strategies_dataframe

        # list of model input features (all signal except random and optimum)
        self.x_columns = [f'{strategy}_signal' for strategy in StockStrategies.strategies_name
                          if strategy not in ('optimum', 'random')]
        # output y (optimum: real data y_true)
        self.y_columns = ['optimum_signal']

        self.x_train = self.x_test = self.y_train = self.y_test = None

        # prepare data before training
        self._data_preprocess()

    def _data_preprocess(self):
        """ Preapare data breofre training """
        # substitute NaN (ie. strategy not ready yet => do nothing
        self.dataset_dataframe = self.dataset_dataframe.fillna('hold')
        # slice columns dataset
        self.dataset_dataframe = self.dataset_dataframe.loc[:, self.x_columns + self.y_columns]

        # Convert 'buy', 'sell' and 'hold' in numeric value
        # TODO: try OneHotEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(list(self.dataset_dataframe[self.y_columns[0]].values) + ['hold'])

        for column in self.dataset_dataframe.columns:
            self.dataset_dataframe[column] = label_encoder.transform(self.dataset_dataframe[column])

        # split features from output
        x_dataframe = self.dataset_dataframe.loc[:, self.x_columns]
        y_dataframe = self.dataset_dataframe.loc[:, self.y_columns]

        # generate TrainingSet and TestSet
        test_size = 1 - self.config['train_test_split']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_dataframe.values,
                                                                                y_dataframe.values,
                                                                                test_size=test_size,
                                                                                shuffle=True)

    def mlp_predict(self):
        """ Use an mlp to choose the best strategy """
        hidden_layer_sizes = self.config['mlp']['hidden_layer_sizes']
        classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000)

        classifier.fit(self.x_train, self.y_train.ravel())
        y_pred = classifier.predict(self.x_test)

        return y_pred

    def random_forest_predict(self):
        """ Use random forest to choose the best strategy """
        n_estimators = self.config['random_forest']['n_estimators']
        classifier = RandomForestClassifier(n_estimators=n_estimators)

        classifier.fit(self.x_train, self.y_train.ravel())
        y_pred = classifier.predict(self.x_test)

        return y_pred
