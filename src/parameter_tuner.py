"""

DeepAR Forecaster using Bayesian-optimization as hyper-parameter tuner.

Environment: tested on Google colab's gpu runtime environment. Expected to also work on cpu environment.
-> works well on cpu.

Used 3rd party packages: pandas, numpy, mxnet, gluonts

Used internal packages: typing, os, pathlib, warnings

Usage:

    1. prepare a dataset with timestamp as index.

    2. split the dataset into train and valid set.
       The types of two datasets are recommended as pandas DataFrame. May not work on other types.

    3. set the parameter bounds as dictionary with tuples or single number.
        ex) {num_cells : (20, 40), epochs : 30, ... }
        The parameters used are defined in the class DeepAR.model method

                                                    Coded by Won Jun Kim
"""
import warnings
from pathlib import Path
from typing import Dict, Optional, List
import mxnet as mx
import os
import pandas as pd
from bayes_opt import BayesianOptimization
from gluonts.evaluation.backtest import make_evaluation_predictions

from gluonts.model.predictor import Predictor
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from tqdm import tqdm

from src._base import BayesianTuner
from src.feature_pattern import FeatureMaker
from src.utils.utilities import refine_forecasts, plot_prob_forecasts, fill_features, evaluate_on_custom_valid


class DeepVARTuner(BayesianTuner):
    def __init__(self,
                 train_df: pd.DataFrame,
                 valid_df: pd.DataFrame,
                 learning_rate: float,
                 target_feature_name: str,
                 features: Optional[List[str]] = None,
                 prediction_length: int = 48 * 2,
                 batch_size: int = 32,
                 ):
        super().__init__(train_df, valid_df)

        # check available device
        if not mx.test_utils.list_gpus():
            self.ctx = mx.context.cpu()

        else:
            self.ctx = mx.context.gpu()

        self.prediction_length = prediction_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.internal_iter_num = 0
        self.current_saving_folder = None
        self.features = features
        self.feature_finder = None

        self.target_feature_name = target_feature_name
        self.features.insert(0, target_feature_name)

        self.train_ds = self.transform_to_ListData(train_df, input_columns=self.features)
        self.valid_ds = self.transform_to_ListData(valid_df, input_columns=self.features)

    # method to convert given dataframe into ListData class of gluonts
    def transform_to_ListData(self,
                              dataframe: pd.DataFrame,
                              input_columns: Optional[List] = None,
                              ):
        target_list = []

        if input_columns is not None:
            for col in input_columns:
                target_list.append(dataframe[col].values[:-self.prediction_length])

        dataset = ListDataset(
            [{"start": dataframe.index[0],
              "target": target_list
              }],
            freq='30min',
            one_dim_target=False
        )

        return dataset

    # calculate the sum of quantile loss from given period
    # quantile forecast values and true values must be entered. Prediction is made in this method
    def forecast_quantiles(self,
                           dataset: ListDataset,
                           predictor: Predictor,
                           num_samples: int = 200) -> (pd.DataFrame, pd.DataFrame):
        quantile_forecasts = {}
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        forecast_iter, ts_iter = make_evaluation_predictions(dataset, predictor=predictor, num_samples=num_samples)

        forecasts = next(forecast_iter)  # forecasts: instance from SampleForecast

        for quantile in quantiles:
            # forecast.quantile: np.NdArray shaped (pred_len, num_feature)
           quantile_forecasts[quantile] = forecasts.quantile(q=quantile)[:, 0]

        start_date = dataset.list_data[0]['start']
        date_range = pd.date_range(start_date, periods=48*9, freq='30min')
        result = pd.DataFrame(quantile_forecasts, columns=quantiles, index=date_range[-self.prediction_length:])
        return result

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
            'epochs': int(epochs)
        }

        trainer = Trainer(epochs=int(epochs),
                          batch_size=self.batch_size, patience=50,
                          ctx=self.ctx,
                          learning_rate=self.learning_rate)

        estimator = DeepVAREstimator(trainer=trainer, freq='30min', target_dim=len(self.features),
                                     prediction_length=self.prediction_length, **estimator_params)

        predictor = estimator.train(training_data=self.train_ds)

        forecast_df = self.forecast_quantiles(self.valid_ds, predictor, 10)
        forecast_df = refine_forecasts(forecast_df)
        y_true = self.valid_df.TARGET[-self.prediction_length:]
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
                   verbose: int = 2,
                   init_points: int = 4,
                   n_iter: int = 20,
                   saving_folder: Optional[str] = None,
                   plot: bool = True,
                   num_samples: int = 200,
                   **kwargs):
        self.epochs = kwargs['epochs']

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

            estimator = DeepVAREstimator(deepAR.max['params'], trainer=trainer, freq='30min',
                                         prediction_length=self.prediction_length,
                                         target_dim=len(self.features),
                                         cell_type='lstm')
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
            estimator = DeepVAREstimator(trainer=trainer, freq='30min',
                                         prediction_length=self.prediction_length,
                                         target_dim=len(self.features), cell_type='lstm', **kwargs)

            predictor = estimator.train(training_data=self.train_ds)

            self._estimator = estimator
            self._predictor = predictor

        print("\nEvaluating on valid set...")
        loss = self.evaluate(self.valid_ds, self.valid_df,
                             predictor, best_params,
                             saving_folder=saving_folder, num_samples=num_samples,
                             plot=plot)

        self._valid_loss = round(loss, 4)

    # evaluate on given dataset and dataframe and returns the loss
    def evaluate(self,
                 dataset: ListDataset,
                 original_dataframe: pd.DataFrame,
                 predictor,
                 best_params,
                 saving_folder: str,
                 num_samples: int = 200,
                 plot=True) -> float:

        forecast_df = self.forecast_quantiles(dataset, predictor, num_samples)

        refined_forecast_df = refine_forecasts(forecast_df)

        y_true = original_dataframe[self.target_feature_name][-self.prediction_length:]
        quantile_sum = self.quantile_loss(y_true.values, refined_forecast_df)

        print(f'The lowest sum of quantile loss for validation set is {round(quantile_sum, 4)}'
              f' with parameters {best_params}')

        # Model saving process
        directory_name = f'/model_{str(round(quantile_sum, 4))}_epoch_{self.epochs}'
        if not saving_folder:
            curpath = os.getcwd()
            saving_folder = curpath + '/saved_model' + directory_name

        else:
            saving_folder = saving_folder + directory_name

        self.current_saving_folder = saving_folder

        # model saving
        try:
            print("Saving the model under " + saving_folder + " with records.")
            Path(saving_folder).mkdir(parents=True)
            predictor.serialize(Path(saving_folder))

            # to loads it back,
            # from gluonts.model.predictor import Predictor
            # predictor_deserialized = Predictor.deserialize(Path("/tmp/"))

            if self.records:
                record = self.return_records()
                record.to_csv(saving_folder + '/optimizer_record.csv')

            forecast_df.to_csv(saving_folder + '/before_refine_forecast.csv')
            refined_forecast_df.to_csv(saving_folder + '/refined_forecast.csv')
            print("Successfully saved model.")

        except FileExistsError:
            warnings.warn(f"File or directory already exists in " + saving_folder)

        except:
            warnings.warn("Saving file failed due to unknown reason."
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
                        num_samples: int = 200,
                        test_start: int = 0,
                        test_end: int = 81,
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
        all_quantile_forecasts = []
        filename = f'/test_{fill_method}.csv'
        test_file_path = test_path + filename

        if not Path(test_file_path).is_file():
            if not self.feature_finder and isinstance(fill_method, str) and fill_method[:7] == 'pattern':
                finder = FeatureMaker(test_path, timestamped_train_dir="../data/Timestamped.csv",
                                      test_end=test_end, feature_columns=['DHI', 'DNI', 'WS', 'RH', 'T'])
                self.feature_finder = finder

            test_df = fill_features(test_path=test_path, fill_method=fill_method, test_start=test_start,
                                    test_end=test_end, finder=self.feature_finder)

        else:
            test_df = pd.read_csv(test_file_path, index_col=0)

        if fill_method != 'pattern':
            test_df = test_df[self.features]

        for file_num in tqdm(range(test_start, test_end), desc='Predicting on each test dataset'):
            temp = test_df.iloc[file_num*48*9:(file_num+1)*48*9, :]
            test_ds = self.transform_to_ListData(temp, input_columns=self.features)

            forecast_df = self.forecast_quantiles(test_ds, self._predictor, num_samples)
            forecast_df = refine_forecasts(forecast_df)
            forecast_df = pd.DataFrame(forecast_df.values, columns=forecast_df.columns)

            all_quantile_forecasts.append(forecast_df)

        final = pd.concat(all_quantile_forecasts, axis=0)
        self.save_submission(final,  fill_method=fill_method, **kwargs)

        return final

    def save_submission(self, target_df: pd.DataFrame,
                        fill_method: str,
                        for_valid: bool = False,
                        sample_submission_file_path: Optional[str] = None,
                        ) -> None:
        cur_path = os.getcwd()
        if not sample_submission_file_path:
            sample_submission_file_path = cur_path + '/sample_submission.csv'

        try:
            sample_sub = pd.read_csv(sample_submission_file_path, index_col=0)
            target_df.index = sample_sub.index[:target_df.shape[0]]
            target_df.columns = sample_sub.columns

        except FileNotFoundError:
            warnings.warn(f"Could not read sample submission file from {sample_submission_file_path}.")

        except:
            warnings.warn(f"Index or column could not be set on target df from sample submission file.")

        try:
            if for_valid:
                submission_file_name = f'/submission_{self.return_best_loss()}_{fill_method}_valid.csv'

            else:
                submission_file_name = f'/submission_{self.return_best_loss()}_{fill_method}_dacon.csv'

            target_df.to_csv(self.current_saving_folder + submission_file_name)
            print("Submission file is successfully save into " + self.current_saving_folder)

        except:
            warnings.warn(f"Could not save the submission file to {self.current_saving_folder}")


def save_result_as_csv(loss, result, model_dir):
    loss_df = pd.DataFrame.from_dict(loss, orient='columns')
    loss_df.index = ['sum']
    result_T = pd.concat(result, axis=1)
    merged = pd.concat([loss_df, result_T], axis=0)
    merged.to_csv(model_dir + f'/custom_valid_figures/result.csv')
    return merged


if __name__ == '__main__':
    import mxnet
    import numpy as np

    np.random.seed(0)
    mxnet.random.seed(0)

    df = pd.read_csv("../data/Timestamped.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index(df['Timestamp'])

    train_end = pd.to_datetime('2017-12-31 23:30:00')
    temp = df[['TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    train = temp[:train_end]

    valid_start = pd.to_datetime('2018-12-14 00:00:00')
    valid_end = pd.to_datetime('2018-12-22 23:30:00')
    valid = temp[valid_start:valid_end]

    valid_base = df.copy()
    valid_base.drop('Timestamp', axis=1, inplace=True)

    def run(params, lr, test_end, test=True):
        saving_folder_path = '../saved_model'
        sample_submission_file = '../data/submission/sample_submission.csv'
        test_file_path = '../data/test'

        valid_dir = '../data/valid'

        features = ['DHI', 'DNI', 'WS', 'RH', 'T']

        tuner = DeepVARTuner(train, valid, learning_rate=lr, features=features,
                             target_feature_name='TARGET', batch_size=16)

        tuner.tune_model(plot=False,
                         saving_folder=saving_folder_path,
                         **params)

        submission = tuner.predict_on_test(test_path=valid_dir, test_end=test_end, for_valid=True,
                                           sample_submission_file_path=sample_submission_file)

        methods = [None]

        loss = {}
        result = []
        model_dir = tuner.current_saving_folder

        for method in methods:
            true_df_dir = '../data/valid/original/timestamped'
            forecasted_df_name = f'submission_{tuner.return_best_loss()}_{method}_valid.csv'

            total_loss, result_df = evaluate_on_custom_valid(true_df_dir, model_dir, forecasted_df_name,
                                                             method=method, show_plot=False, test_end=test_end,
                                                             sub_plot_row=5,
                                                             sub_plot_col=4)
            loss[method] = [total_loss]
            result.append(result_df)

            if method:
                result_df.to_csv(model_dir + f'/custom_valid_figures/{method}_result.csv', index=False)

            else:
                result_df.to_csv(model_dir + f'/custom_valid_figures/result.csv', index=False)

        save_result_as_csv(loss, result, model_dir)

        if test:
            submission = tuner.predict_on_test(test_path=test_file_path, feat_dynamic_real=features,
                                               test_end=test_end, for_valid=False,
                                               sample_submission_file_path=sample_submission_file)

    test_end = 20
    # make_validation_set(valid_base, 1, 12, test_end=test_end)

    params = {'epochs': 5,  # 20
              'context_length': 105,
              'num_cells': 47,
              'num_layers': 4}
    features = ['DHI', 'DNI', 'WS', 'RH', 'T']

    saving_folder_path = '../saved_model'
    sample_submission_file = '../data/submission/sample_submission.csv'
    test_dir = '../data/test'
    valid_dir = '../data/valid'

    tuner = DeepVARTuner(train, valid, learning_rate=0.01, features=features,
                         target_feature_name='TARGET', batch_size=16)

    tuner.tune_model(plot=True, saving_folder=saving_folder_path, **params)

    submission = tuner.predict_on_test(test_path=valid_dir, test_end=test_end, for_valid=True,
                                       sample_submission_file_path=sample_submission_file)

    methods = [None]

    loss = {}
    result = []
    model_dir = tuner.current_saving_folder

    for method in methods:
        true_df_dir = '../data/valid/original/timestamped'
        forecasted_df_name = f'submission_{tuner.return_best_loss()}_{method}_valid.csv'

        total_loss, result_df = evaluate_on_custom_valid(true_df_dir, model_dir, forecasted_df_name,
                                                         method=method, show_plot=False, test_end=test_end,
                                                         sub_plot_row=5,
                                                         sub_plot_col=4)
        loss[method] = [total_loss]
        result.append(result_df)

        result_df.to_csv(model_dir + f'/custom_valid_figures/{method}_result.csv', index=False)

    save_result_as_csv(loss, result, model_dir)

    submission = tuner.predict_on_test(test_path=test_dir, test_end=test_end, for_valid=False,
                                       sample_submission_file_path=sample_submission_file)

