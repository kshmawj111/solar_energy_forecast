import pandas as pd
from typing import *
import matplotlib.pyplot as plt
import warnings

# from EDA, we know that some time in a day records target 0 always.
# making that times to zero happens in this method.
# plus, sometimes forecast predicts negative values, so we set them as zero
def refine_forecasts(forecasts: pd.DataFrame, ) -> pd.DataFrame:
    # if index is Datetime object
    if isinstance(forecasts.index, pd.DatetimeIndex):
        zero_dhi_list = [x for x in range(0, 5)] + [x for x in range(20, 24)]

        for row in forecasts.index:
            hour = row.hour

            if hour in zero_dhi_list:
                forecasts.loc[row, :] = 0

            elif row.hour == 19 and row.minute == 30:
                forecasts.loc[row, :] = 0

        return forecasts

    # if not, then reset the column and move the previous string of file names to a new column named 'id'
    else:
        zero_dhi_list = [f'_{x}' for x in range(0, 5)] + [f'{x}' for x in range(20, 24)]

        assert isinstance(forecasts.loc[0, 'id'], str), "Different format of dataframe is given. Have a check."

        for row in forecasts.index:
            time = forecasts.loc[row, 'id'][-6:]

            if time[:2] in zero_dhi_list:
                forecasts.iloc[row, 1:] = 0

            elif time == '19h30m':
                forecasts.iloc[row, 1:] = 0

        return forecasts.set_index('id')


# calculate moving average with given dataframe
def calculate_moving_average(df: pd.DataFrame,
                             windows_size: int = 2,
                             prediction_period: int = 2
                             ) -> pd.DataFrame:
    temp = df.copy(deep=True)

    # moving average for the length of prediction periods
    # previously calculated moving average value is treated as ground truth for the next window calculation
    for day in range(prediction_period):
        ma = temp.rolling(windows_size).mean()

        ma['Day'] = day + 7 + 1
        ma['TARGET'] = 0

        temp = temp.append(ma.iloc[-1, :])

    return temp


# add feature values (DHI, DNI etc..) using moving average method.
# returns a dataframe of all test table with calculated moving average
def make_features(target_dataframe: pd.DataFrame) -> pd.DataFrame:
    test_df = []

    sorted_df = target_dataframe.sort_values(['Hour', 'Minute'], ascending=True)

    parts = []
    for df_num in range(0, 48):
        part_of_df = sorted_df.iloc[df_num * 7:(df_num + 1) * 7, ]
        parts.append(calculate_moving_average(part_of_df))

    test_df.append(pd.concat(parts))
    result = pd.concat(test_df)
    result = result.sort_values(['Day', 'Hour', 'Minute'], ascending=True)

    return result

def plot_prob_forecasts(
                        y_true: Union[pd.DataFrame, pd.Series],
                        quantile_forecasts: Union[dict, pd.DataFrame],
                        saving_path: Optional[str] = None,
                        prediction_length=96,
                        return_forecasts=True
                        ):

    if isinstance(quantile_forecasts, pd.DataFrame):
        quantile_forecasts.index = y_true.index[-prediction_length:]

    elif isinstance(quantile_forecasts, dict):
        quantile_forecasts = pd.DataFrame(quantile_forecasts, index=y_true.index[-prediction_length:])

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    y_true.plot(ax=ax, linewidth=4, linestyle='dashed')  # plot the history target
    plt.legend('True')
    quantile_forecasts.plot(y=quantile_forecasts.columns, ax=ax, linestyle='solid')

    plt.grid(which="both")
    plt.show()

    if saving_path is not None:
        try:
            plt.savefig(saving_path+'/valid_ds_result.png', dpi=700)
            print("Successfully saved the figure")
        except:
            warnings.warn("Saving figure is failed.")

    if return_forecasts:
        return quantile_forecasts