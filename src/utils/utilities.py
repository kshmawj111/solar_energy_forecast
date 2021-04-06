import pandas as pd
import numpy as np
from typing import Union, Optional, List
import matplotlib.pyplot as plt
import warnings
from .timestamper import Timestamper
from tqdm import tqdm
from pathlib import Path
from src.feature_pattern import FeatureMaker


# from EDA, we know that some time in a day records target 0 always.
# making that times to zero happens in this method.
# plus, sometimes forecast predicts negative values, so we set them as zero
def refine_forecasts(forecasts: pd.DataFrame, ) -> pd.DataFrame:
    copied = forecasts.copy(deep=True)

    # if index is Datetime object
    if isinstance(forecasts.index, pd.DatetimeIndex):
        zero_dhi_list = [x for x in range(0, 5)] + [x for x in range(20, 24)]

        for row in copied.index:
            hour = row.hour

            if hour in zero_dhi_list:
                copied.loc[row, :] = 0

            elif row.hour == 19 and row.minute == 30:
                copied.loc[row, :] = 0

    # if not, then reset the column and move the previous string of file names to a new column named 'id'
    else:
        zero_dhi_list = [f'_{x}' for x in range(0, 5)] + [f'{x}' for x in range(20, 24)] # times which dhi is zero always

        assert isinstance(copied.loc[0, 'id'], str), "Different format of dataframe is given. Have a check."

        for row in copied.index:
            time = copied.loc[row, 'id'][-6:]

            if time[:2] in zero_dhi_list:
                copied.iloc[row, 1:] = 0

            elif time == '19h30m':
                copied.iloc[row, 1:] = 0

        copied = copied.set_index('id')

    copied[copied < 0] = 0
    return copied


def make_validation_set(temp_df, min_month: int, max_month: int, test_start=0, test_end=81):
    from pathlib import Path
    import shutil

    save_dir = './data/valid'
    original_dir = './data/valid/original'

    temp = temp_df.copy()

    if not Path(save_dir).is_dir():
        Path(save_dir).mkdir(parents=True)

    else:
        shutil.rmtree(save_dir)

    if not Path(original_dir).is_dir():
        Path(original_dir).mkdir(parents=True)

    def get_month_day():
        import random
        day_for_month = {1: 31, 2: 28, 3:31, 4:30, 5: 31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

        month = random.randint(min_month, max_month)
        day = random.randint(1, 22)

        while day+8 > day_for_month[month]:
            day = random.randint(1, 22)

        month = str(month).zfill(2)
        day = str(day).zfill(2)
        valid_start = f'2018-{month}-{day} 00:00:00'
        valid_end = f'2018-{month}-{int(day) + 8} 23:30:00'

        return valid_start, valid_end

    valid_set_candidates = {}

    while len(valid_set_candidates) <= test_end:
        s, e = get_month_day()

        if s not in valid_set_candidates.keys():
            valid_set_candidates[s] = e

    num = 0
    for k, v in valid_set_candidates.items():
        valid_df = temp[k:v]
        valid_df = valid_df.drop(['Year', 'Month'], axis=1)

        for x in range(9):
            valid_df.iloc[x*48:(x+1)*48 , 0] = x

        valid_df.to_csv(original_dir + f'/{num}.csv', index=False)
        as_test = valid_df.iloc[:48*7, ]
        as_test.to_csv(save_dir + f'/{num}.csv', index=False)
        num += 1

    stamper = Timestamper(original_dir, test_start=test_start, test_end=test_end)
    stamper.stamp()


def plot_prob_forecasts(
                        y_true: Union[pd.DataFrame, pd.Series],
                        quantile_forecasts: Union[dict, pd.DataFrame],
                        saving_path: Optional[str] = None,
                        prediction_length=96,
                        return_forecasts=True,
                        show_plot: bool = True,
                        ):
    quantile_forecasts_copied = None

    if isinstance(quantile_forecasts, pd.DataFrame):
        quantile_forecasts.index = y_true.index[-prediction_length:]
        quantile_forecasts_copied = quantile_forecasts.copy(deep=True)

    elif isinstance(quantile_forecasts, dict):
        quantile_forecasts_copied = pd.DataFrame(quantile_forecasts, index=y_true.index[-prediction_length:])

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    y_true.plot(ax=ax, linewidth=4, linestyle='dashed')  # plot the history target
    plt.legend('True')
    quantile_forecasts_copied.plot(y=quantile_forecasts_copied.columns, ax=ax, linestyle='solid')
    plt.grid(which="both")

    if saving_path is not None:
        try:
            plt.savefig(saving_path+'/valid_ds_result.png', dpi=500)
            print("Successfully saved the figure")
        except:
            warnings.warn("Saving figure is failed.")

    if show_plot:
        plt.show()

    if return_forecasts:
        return quantile_forecasts_copied


# sort df
def sort_columns(sample_df: pd.DataFrame):
    temp = sample_df.copy(deep=True)
    result = pd.DataFrame()

    for col in temp.columns:
        t = temp[col]
        t = t.sort_values(ascending=True)
        t = t.reset_index(drop=True)
        result[col] = t

    return result


# find quantile row
def parse_from_sample(sample_df: pd.DataFrame, quantile: float):
    quantile_index = sample_df.shape[0] * quantile - 1

    result = sample_df.loc[quantile_index, :].values
    return result


# abstract class for Bayesian Tuner
def quantile_loss(y_true: Union[np.ndarray, pd.Series, pd.DataFrame],
                  y_forecast: pd.DataFrame,
                  quantiles: Optional[List[float]] = None):
    y_forecast_copied = y_forecast.copy(deep=True)

    # TODO: may modify the base calculation structure base to numpy instead of dataframe, for performance.
    # default quantiles
    if not quantiles:
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # check if the quantiles are same
    assert len(quantiles) == y_forecast_copied.shape[1], "Number of quantiles from forecast and quantiles list is different"

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


def evaluate_on_custom_valid(timestamped_true_df_dir,
                             model_dir: str,
                             forecasted_df_name: str,
                             method: str=None,
                             test_start=0,
                             test_end=81,
                             sub_plot_row=6,
                             sub_plot_col=3,
                             show_plot=True):
    forecasted_df_dir = model_dir + '/' + forecasted_df_name
    forecasted_df = pd.read_csv(forecasted_df_dir, index_col=0)
    fig, axs = plt.subplots(nrows=sub_plot_row, figsize=(18, 18),
                            ncols=sub_plot_col)

    _ax = None
    _loss = 0
    result = {}

    for file_num in range(test_start, test_end):
        forecast_part = forecasted_df.iloc[file_num * 48 * 2: (file_num + 1) * 48 * 2, ].copy()
        true_df_file_dir = timestamped_true_df_dir + f'/{file_num}.csv'

        true_df = pd.read_csv(true_df_file_dir)
        true_df['Timestamp'] = pd.to_datetime(true_df['Timestamp'])
        true_df = true_df.set_index('Timestamp')
        y_true = true_df['TARGET'][48 * 7:48 * 9]

        loss = quantile_loss(y_true, forecast_part)
        _loss = round(loss, 4)
        result[file_num] = [loss]


        ax = axs[file_num // sub_plot_col, file_num % sub_plot_col]
        ax.set_title(f'{file_num}.csv, loss: {_loss}')
        _ax = ax

        forecast_part.index = y_true.index
        quantile_columns = forecast_part.columns

        forecast_part['True'] = y_true

        forecast_part[quantile_columns].plot(ax=ax, colormap='GnBu',
                                             legend=False, xlabel='',
                                             )
        forecast_part['True'].plot(ax=ax, linewidth=3, xlabel='',
                                   legend=False)


    axLine, axLabel = _ax.get_legend_handles_labels()
    fig.legend(axLine, axLabel, loc='right')
    plt.subplots_adjust(top=0.928,
                        bottom=0.022,
                        left=0.027,
                        right=0.981,
                        hspace=0.46,
                        wspace=0.149)

    plot_save_path = model_dir + f'/custom_valid_figures'

    if not method:
        fig.suptitle(f'Validation result for {method}')
        save_file_name = f'valid_result_{method}.png'

    else:
        fig.suptitle(f'Validation result')
        save_file_name = f'valid_result.png'

    if not Path(plot_save_path).is_dir():
        Path(plot_save_path).mkdir()

    plt.savefig(plot_save_path + f'/{save_file_name}')

    if show_plot:
        plt.show()

    result_df = pd.DataFrame.from_dict(result, orient='index', columns=[f'{method}'])
    total_loss = result_df.sum().sum()

    return total_loss, result_df
