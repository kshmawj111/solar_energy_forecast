from typing import Union, Optional, List
import numpy as np
import pandas as pd
import warnings

try:
    from bayes_opt import BayesianOptimization

except ImportError as e:
    print("Bayesian Optimization package cannot be imported. Check if it is installed."
          "If not installed use $ pip install bayesian-optimization")
    exit(-10)


# abstract class for Bayesian Tuner
# DeepAR, DeepVAR model inherits this abstract class
# and they must implement model and tune model methods
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
        self.records = []  # record is for saving each iteration result in the model training

    # calculate the sum of quantile loss from given forecast
    def quantile_loss(self, y_true: Union[np.ndarray, pd.Series, pd.DataFrame],
                      y_forecast: pd.DataFrame,
                      quantiles: Optional[List[float]] = None):
        c_y_forecast = y_forecast.copy(deep=True)

        # TODO: may modify the base calculation structure base to numpy instead of dataframe, for performance.
        # default quantiles
        if not quantiles:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # check if the quantiles are same
        assert len(quantiles) == c_y_forecast.shape[1], "Number of quantiles from forecast and quantiles list is different"

        if isinstance(c_y_forecast, pd.DataFrame):
            c_y_forecast.columns = quantiles

        # cast y_true
        if isinstance(y_true, pd.DataFrame):
            assert y_true.shape[1] != 1, "y_true value must be shape of ( , 1)"
            y_true = y_true.values

        elif isinstance(y_true, pd.Series):
            y_true = y_true.values

        # quantile loss = max(q*(y_pred - y_true), (1-q)*(y_pred - y_true)) q: quantile value
        for quantile in c_y_forecast.columns:
            loss = c_y_forecast[quantile] - y_true  # forcast: (1, predicton_length), y_true: scala value,

            # as loss is pd.Series type, to use max() we need to apply max() lambda function to all the rows
            # but this is very expensive op. Therefore, below code is applied
            loss = np.where(loss < 0, -loss * (1 - quantile), quantile * loss)

            c_y_forecast[quantile] = loss

        return c_y_forecast.sum().sum()

    # abstract method
    def model(self, **kwargs):
        raise NotImplementedError

    def tune_model(self, **kwargs):
        raise NotImplementedError

    # getter
    def return_records(self) -> pd.DataFrame:
        if self.records:
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