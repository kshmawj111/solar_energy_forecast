"""

DeepAR hyper-parameter tuner using Bayesian-optimization.

Environment: tested on Google colab's gpu runtime environment. Expected to also work on cpu environment

Used 3rd party packages: pandas, numpy, mxnet, gluonts

Used internal packages: typing, os, pathlib, warnings

Usage:

    1. prepare a dataset with timestamp as index.

    2. split the dataset into train and valid set.
       The types of two datasets are recommended as pandas DataFrame. May not work on other types.

    3. set the parameter bounds as dictionary with tuples or single number.
        ex) {num_cells : (20, 40), epochs : 30, ... }
        The parameters used are defined in the class DeepAR.model method

"""
import pandas as pd
import numpy as np
from typing import Dict, List
import mxnet as mx
import os
from pathlib import Path
import warnings

from tqdm import tqdm
from gluonts.mx.distribution import DistributionOutput, StudentTOutput

from utils.timestamper import Timestamper
from utils.utilities import *

from gluonts.model.predictor import Predictor
from gluonts.dataset.common import Dataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions

try:
    from bayes_opt import BayesianOptimization

except ImportError as e:
    print("Bayesian Optimization package cannot be imported. Check if it is installed."
          "If not installed use $ pip install bayesian-optimization")
    exit(-10)


# abstract class for Bayesian Tuner
class BayesianTuner:
    def __init__(self,
                 # input dataset format not determined
                 train_df,
                 valid_df
                 ):  # {param_name: (lower, upper), ... }
        self._valid_loss = None
        self._predictor = None
        self._estimator = None
        self.train_df = train_df
        self.valid_df = valid_df
        self.records = []

    # given forecast and true values, return the sum of all
    def quantile_loss(self, y_true: Union[np.ndarray, pd.Series, pd.DataFrame],
                      y_forecast: pd.DataFrame,
                      quantiles: Optional[List[float]] = None):
        y_forecast_copied = y_forecast.copy(deep=True)

        # TODO: may modify the base calculation structure base to numpy instead of dataframe, for performance.
        # default quantiles
        if not quantiles:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # check if the quantiles are same
        assert len(quantiles) == y_forecast_copied.shape[
            1], "Number of quantiles from forecast and quantiles list is different"

        if isinstance(y_forecast_copied, pd.DataFrame):
            y_forecast_copied.columns = quantiles

        # cast y_true
        if isinstance(y_true, pd.DataFrame):
            assert y_true.shape[1] != 1, "y_true value must be shape of ( , 1)"
            y_true = y_true.values

        elif isinstance(y_true, pd.Series):
            y_true = y_true.values

        # quantile loss = max(q*(y_pred - y_true), (1-q)*(y_pred, y_true))
        for quantile in y_forecast_copied.columns:
            loss = y_forecast_copied[quantile] - y_true  # forcast: (1, predicton_length), y_true: scala value
            loss = np.where(loss < 0, -loss * (1 - quantile), quantile * loss)

            y_forecast_copied[quantile] = loss

        return y_forecast_copied.sum().sum()

    # abstract method
    def model(self, **kwargs):
        raise NotImplementedError

    def tune_model(self, **kwargs):
        raise NotImplementedError

    # exception handling
    def return_records(self) -> pd.DataFrame:
        if self.records is not None:
            return pd.concat(self.records)

        else:
            warnings.warn("You must first train the model to get trained estimator. Call tune_model first.")

    def return_estimator(self):
        if self._estimator is not None:
            return self._estimator

        else:
            warnings.warn("You must first train the model to get trained estimator. Call tune_model first.")

    def return_predictor(self):
        if self._predictor is not None:
            return self._predictor

        else:
            warnings.warn("You must first train the model to get trained predictor. Call tune_model first.")

    def return_best_loss(self):
        if self._valid_loss != 0:
            return round(self._valid_loss, 4)

        else:
            warnings.warn("You must first train the model to get best loss. Call tune_model first.")


class DeepARTuner(BayesianTuner):
    def __init__(self,
                 train_df: pd.DataFrame,
                 valid_df: pd.DataFrame,
                 learning_rate: float,
                 target_feature_name: str,
                 feat_dynamic_features: Optional[List[str]] = None,
                 prediction_window: int = 48*2,
                 batch_size: int = 32,
                 ):
        super().__init__(train_df, valid_df)

        # check available device
        if not mx.test_utils.list_gpus():
            self.ctx = mx.context.cpu()

        else:
            self.ctx = mx.context.gpu()

        self.prediction_window = prediction_window
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.internal_iter_num = 0
        self.current_saving_folder = None
        self.feat_dynamic_columns = feat_dynamic_features

        if feat_dynamic_features is not None:
            self.use_feat_dynamic_real = True

        else:
            self.use_feat_dynamic_real = False

        self.target_feature_name = target_feature_name
        self.train_ds = self.transform_to_ListData(train_df, for_train=True, feat_dynamic_real=feat_dynamic_features)
        self.valid_ds = self.transform_to_ListData(valid_df, for_train=False, feat_dynamic_real=feat_dynamic_features)

    # method to convert given dataframe into ListData class of gluonts
    """def transform_to_ListData(self):
        feat_dynamic_real_train = []
        feat_dynamic_real_valid = []

        if self.use_feat_dynamic_real:
            for col in self.feat_dynamic_columns:
                feat_dynamic_real_train.append(self.train_df[col].values[:-self.prediction_window])
                feat_dynamic_real_valid.append(self.valid_df[col].values)

        self.train_ds = ListDataset(
            [{"start": self.train_df.index[0],
              "target": self.train_df[self.target_feature_name].values[:-self.prediction_window],
              "feat_dynamic_real": feat_dynamic_real_train
              }],
            freq="30min",
            one_dim_target=True
        )

        valid_target = self.valid_df[self.target_feature_name].values[:-self.prediction_window]
        valid_target = np.append(valid_target, np.zeros((48*2, )))

        self.valid_ds = ListDataset(
            [{"start": self.valid_df.index[0],
              "target": valid_target,
              "feat_dynamic_real": feat_dynamic_real_valid
              }],
            freq="30min",
            one_dim_target=True
        )"""

    def transform_to_ListData(self,
                              dataframe: pd.DataFrame,
                              for_train: bool,
                              feat_dynamic_real: Optional[List] = None,
                              ):
        feat_dynamic_real_values = []

        if feat_dynamic_real is not None:
            if for_train:
                for col in feat_dynamic_real:
                    feat_dynamic_real_values.append(dataframe[col].values[:-self.prediction_window])

            else:
                for col in feat_dynamic_real:
                    feat_dynamic_real_values.append(dataframe[col].values)

        if for_train is True:
            target = dataframe[self.target_feature_name].values[:-self.prediction_window]

        else:
            target = dataframe[self.target_feature_name].values

        dataset = ListDataset(
            [{"start": dataframe.index[0],
              "target": target,
              "feat_dynamic_real": feat_dynamic_real_values
              }],
            freq='30min',
            one_dim_target=True
        )
        return dataset

    # calculate the sum of quantile loss from given period
    # quantile forecast values and true values must be entered. Prediction is made in this method
    def forecast_quantiles(self,
                           dataset: Dataset,
                           predictor: Predictor,
                           num_samples: int = 1000) -> (pd.DataFrame, pd.DataFrame):
        quantile_forecasts = {}
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        forecast_iter, ts_iter = make_evaluation_predictions(dataset, predictor=predictor, num_samples=num_samples)

        forecasts = next(forecast_iter)  # forecasts: instance from SampleForecast
        tss = next(ts_iter)  # tss : instance from pd.DataFrame

        samples = pd.DataFrame(forecasts.samples)
        samples = sort_columns(samples)

        for quantile in quantiles:
            quantile_forecasts[quantile] = parse_from_sample(samples, quantile)

        result = pd.DataFrame(quantile_forecasts, columns=quantiles, index=tss.index[-self.prediction_window:])
        return samples, result

    # deepAR model for bayesian optimization.
    # This returns the sum of quantile loss for given parameters selected by bayesian optimizer
    def model(self,
              epochs,
              context_length,
              num_cells,
              num_layers
              ) -> float:

        estimator_params = {
            'cell_type': 'lstm',
            'context_length': int(context_length),
            'num_cells': int(num_cells),
            'num_layers': int(num_layers),
            'use_feat_dynamic_real': self.use_feat_dynamic_real,
            'epochs': int(epochs)
        }

        trainer = Trainer(epochs=int(epochs),
                          batch_size=self.batch_size,
                          ctx=self.ctx,
                          learning_rate=self.learning_rate)

        estimator = DeepAREstimator(estimator_params, trainer=trainer, freq='30min',
                                    prediction_length=self.prediction_window, distr_output=self.dist_output)

        predictor = estimator.train(training_data=self.train_ds)

        # TODO: maybe backtest_metrics can be used to shorten the process below
        _, forecast_df = self.forecast_quantiles(self.valid_ds, predictor, 1000)
        forecast_df = refine_forecasts(forecast_df)
        y_true = self.valid_df.TARGET[-self.prediction_window:]
        quantile_loss = self.quantile_loss(y_true, forecast_df)

        # record inserting
        iter_record = pd.DataFrame(estimator_params, index=[self.internal_iter_num])
        iter_record['epochs'] = int(epochs)
        iter_record['batch_size'] = self.batch_size
        iter_record['learning_rate'] = round(self.learning_rate, 4)
        iter_record['quantile_loss'] = round(quantile_loss)

        self.internal_iter_num += 1
        self.records.append(iter_record)

        # As bayesian optimizer tries to maximize the target value
        # to make the model work, we need to inverse the sign so that it minimizes the loss
        return -quantile_loss

    # call this method when you actually tune
    def tune_model(self,
                   pbounds: Optional[Dict] = None,
                   dist_output: DistributionOutput = StudentTOutput(),
                   verbose: int = 2,
                   init_points: int = 4,
                   n_iter: int = 20,
                   saving_folder: Optional[str] = None,
                   plot: bool = True,
                   num_samples: int = 1000,
                   **kwargs):
        self.dist_output = dist_output

        if pbounds is not None:
            skip_tune = False

        else:
            skip_tune = True

        # when you want to tune the model
        if not skip_tune:
            print('Tuning start...')
            deepAR = BayesianOptimization(f=self.model, pbounds=pbounds, verbose=verbose)
            deepAR.maximize(init_points=init_points, n_iter=n_iter)
            print('best_target_value:', -deepAR.max['target'])

            trainer = Trainer(epochs=int(deepAR.max['params']['epochs']),
                              batch_size=self.batch_size,
                              ctx=self.ctx,
                              learning_rate=self.learning_rate
                              )

            estimator = DeepAREstimator(deepAR.max['params'], trainer=trainer, distr_output=self.dist_output,
                                        freq='30min', prediction_length=self.prediction_window,
                                        cell_type='lstm', use_feat_dynamic_real=self.use_feat_dynamic_real)
            best_params = deepAR.max["params"]

            predictor = estimator.train(training_data=self.train_ds)

            self._estimator = estimator
            self._predictor = predictor

        # when you do not want to tune the model but train with give parameter(**kwargs)
        else:
            print('Training start...')
            best_params = kwargs
            trainer = Trainer(epochs=int(kwargs['epochs']),
                              batch_size=self.batch_size,
                              ctx=self.ctx,
                              learning_rate=self.learning_rate)

            kwargs.pop('epochs', None)
            estimator = DeepAREstimator(**kwargs, trainer=trainer, distr_output=self.dist_output,
                                        freq='30min', prediction_length=self.prediction_window,
                                        cell_type='lstm', use_feat_dynamic_real=self.use_feat_dynamic_real)

            predictor = estimator.train(training_data=self.train_ds)

            self._estimator = estimator
            self._predictor = predictor

        print("\nEvaluating on valid set...")
        loss = self.evaluate(self.valid_ds, self.valid_df,
                             predictor, best_params,
                             saving_folder=saving_folder_path, num_samples=num_samples,
                             plot=plot)

        self._valid_loss = round(loss, 4)

    # evaluate on given dataset and dataframe and returns the loss
    def evaluate(self,
                 dataset: ListDataset,
                 original_dataframe: pd.DataFrame,
                 predictor,
                 best_params,
                 saving_folder,
                 num_samples: int = 1000,
                 plot=True) -> float:

        samples, forecast_df = self.forecast_quantiles(dataset, predictor, num_samples)

        refined_forecast_df = refine_forecasts(forecast_df)

        y_true = original_dataframe[self.target_feature_name][-self.prediction_window:]
        quantile_sum = self.quantile_loss(y_true.values, refined_forecast_df)

        print(f'The lowest sum of quantile loss for validation set is {round(quantile_sum, 4)}'
              f' with parameters {best_params}')

        # Model saving process
        directory_name = f'/model_{str(round(quantile_sum, 4))}_{self.dist_output.__class__.__name__[0]}'
        if not saving_folder:
            curpath = os.getcwd()
            saving_folder = curpath + '/saved_model' + directory_name

        else:
            saving_folder = saving_folder + directory_name

        # model saving
        try:
            print("Saving the model under " + saving_folder + " with records.")
            Path(saving_folder).mkdir(parents=True)

            self.current_saving_folder = saving_folder
            predictor.serialize(Path(saving_folder))

            # to loads it back,
            # from gluonts.model.predictor import Predictor
            # predictor_deserialized = Predictor.deserialize(Path("/tmp/"))

            if self.records is not None:
                record = self.return_records()
                record.to_csv(saving_folder + '/optimizer_record.csv')

            forecast_df.to_csv(saving_folder + '/before_refine_forecast.csv')
            refined_forecast_df.to_csv(saving_folder + '/refined_forecast.csv')
            samples.to_csv(saving_folder + '/samples.csv')
            print("Successfully saved model.")

        except FileExistsError:
            warnings.warn(f"File or directory already exists in {saving_folder}")

        except:
            warnings.warn("Saving file failed due to unknown reason. "
                          "High probability of collision in predictor serialization is assumed")

        if plot:
            try:
                plot_prob_forecasts(y_true, refined_forecast_df, saving_path=saving_folder)
            except:
                warnings.warn("Cannot plot.")

        return quantile_sum

    def load_predictor(self, predictor_path):
        self._predictor = Predictor.deserialize(predictor_path)

    def predict_on_test(self,
                        test_path: str,
                        fill_method: Optional[str] = None,
                        num_samples: int = 500,
                        feat_dynamic_real: Optional[List[str]] = None,
                        **kwargs) -> pd.DataFrame:
        """

        Parameters
        ----------
         test_path : str => path of directory or folder containing all the test files
         fill_method : str => method for filling in the features for prediction. 'ewm', 'rolling' are allowed at the moement.
         num_samples: int => amount of which to collect for sampling from probability distribution

        Inner work
        ----------
         reads each test csv and converts it to ListDataset.
         If fill_method is not None, then do the job to fill in the features for prediction.


        Returns
        -------
         DataFrame containing quantile forecasts

        """

        # if timestamped csv files for test data under timestamped folder are not available,
        # it creates the new one using the test csv given from Dacon
        test_stamper = Timestamper(test_path=test_path)
        timestamped_path = test_stamper.stamp()
        all_quantile_forecasts = []

        for file_num in range(0, 81):
            current_file = f'/{file_num}.csv'
            test_df = pd.read_csv(timestamped_path + current_file)

            if fill_method is not None:
                test_df = make_features(test_df, method=fill_method)

            test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
            test_df = test_df.set_index(test_df['Timestamp'])

            test_range = pd.date_range(test_df.index[0], periods=48 * 9, freq='30min')
            test_df.index = test_range

            test_ds = self.transform_to_ListData(test_df, for_train=False, feat_dynamic_real=feat_dynamic_real)

            _, forecast_df = self.forecast_quantiles(test_ds, self._predictor, num_samples)
            forecast_df = refine_forecasts(forecast_df)
            forecast_df = pd.DataFrame(forecast_df.values, columns=forecast_df.columns)

            all_quantile_forecasts.append(forecast_df)

        final = pd.concat(all_quantile_forecasts, axis=0)
        self.save_submission(final, **kwargs)

        return final

    def save_submission(self, target_df: pd.DataFrame,
                        sample_submission_file_path: Optional[str] = None) -> None:
        cur_path = os.getcwd()
        if not sample_submission_file_path:
            sample_submission_file_path = cur_path + '/sample_submission.csv'

        try:
            sample_sub = pd.read_csv(sample_submission_file_path, index_col=0)
            target_df.index = sample_sub.index
            target_df.columns = sample_sub.columns

        except:
            warnings.warn(f"Could not read sample submission file from {sample_submission_file_path}.")

        try:
            submission_file_name = f'/submission_{self.return_best_loss()}.csv'
            target_df.to_csv(self.current_saving_folder + submission_file_name)
            print("Submission file is successfully save into " + self.current_saving_folder)

        except:
            warnings.warn(f"Could not save the submission file to {self.current_saving_folder}")


if __name__ == '__main__':
    import mxnet

    np.random.seed(0)
    mxnet.random.seed(0)

    df = pd.read_csv("./data/Timestamped.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index(df['Timestamp'])

    cut_edge = pd.to_datetime('2017-12-31 23:30:00')  # 2년치 학습
    temp = df[['TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    valid_start = pd.to_datetime('2018-12-11 00:00:00')
    valid_end = pd.to_datetime('2018-12-19 23:30:00')

    train = temp[:cut_edge]
    valid = temp[valid_start:valid_end]

    from gluonts.mx.distribution import *

    dist_output = GaussianOutput()

    saving_folder_path = './saved_model'
    sample_submission_file = './data/submission/sample_submission.csv'
    test_file_path = './data/test'

    params = {'epochs': 1, # 20
              'context_length': 105,
              'num_cells': 47,
              'num_layers': 6}

    features = ['DHI', 'DNI', 'WS', 'RH', 'T']
    tuner = DeepARTuner(train, valid, learning_rate=0.001, target_feature_name='TARGET',
                        batch_size=32, feat_dynamic_features=features)
    tuner.tune_model(dist_output=dist_output,
                     saving_folder=saving_folder_path,
                     **params)

    submission = tuner.predict_on_test(test_path=test_file_path, fill_method='rolling', feat_dynamic_real=features,
                                       sample_submission_file_path=sample_submission_file)