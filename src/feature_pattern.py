import pandas as pd
from tqdm import tqdm
from pathlib import Path

"""
    DeepAR is something not compatible with what we're were working at.
    We wanted multi-variate time series regression or forecast and 
    DeepAR partially supports it. 
    
    For DeepAR to forecast future in multi-variate way, we need to give
    future features. (?) This model is best fit with when the feature is ONLY time.
    Therefore, we moved on to DeepVAR which supports multi-variate method as we've aimed to.
    
    This FeatureMaker was written for DeepAR to produce the best feature used in the prediction.
    Its objective is to make the best and similar feature values from history and add them to
    the test dataset we use
    

"""
class FeatureMaker:
    def __init__(self,
                 test_path: str,
                 method: str,
                 timestamped_train_path: str,
                 feature_columns: list,
                 test_end: int,
                 test_start: int = 0,
                 ):
        self.test_files_dir = test_path
        self.timestamped_train_dir = timestamped_train_path
        self.feature_columns = feature_columns
        self.test_start = test_start
        self.test_end = test_end
        self.method = method

        if 'pattern' in method:
            self.searched_result = self.search_similar()

    def return_search_result(self):
        if self.searched_result:
            return self.searched_result

    def search_similar(self) -> dict:
        result = {}

        for file_num in tqdm(range(self.test_start, self.test_end), desc='Finding the most similar features for test set from training set'):
            file_name = f'/{file_num}.csv'
            test_df = pd.read_csv(self.test_files_dir + file_name, index_col=0)
            train_df = pd.read_csv(self.timestamped_train_dir)
            search_result = {}

            for x in range(0, 365*3 - 6):
                compared = train_df.loc[(x*48): (x+7)*48 -1, ].copy()
                compared = compared[self.feature_columns]
                compared = compared.reset_index(drop=True)

                test_df_copied = test_df[self.feature_columns].copy()
                test_df_copied = test_df_copied.reset_index()

                loss_df = compared - test_df_copied

                loss_df = loss_df.apply(lambda row: row.apply(lambda x: x*x))
                loss = loss_df.sum().sum()
                search_result[train_df.loc[x*48, 'Timestamp']] = loss

            search_result = pd.DataFrame.from_dict(search_result, orient='index')
            search_result.columns = [file_name]
            search_result = search_result[file_name].sort_values(ascending=True)
            result[file_name[1:]] = search_result.index[1:]

        self.searched_result = result
        return result

    def get_features_from_train(self, train_df: pd.DataFrame, file_name: str, num_samples: int):
        features = None

        if self.method == 'pattern':
            start_date_idx = train_df.index.get_loc(self.searched_result[file_name][0])
            features = train_df.iloc[start_date_idx + 48 * 7:start_date_idx + 48 * 9, :]

        elif self.method == 'pattern_mean':
            start_date_idx = train_df.index.get_loc(self.searched_result[file_name][0])
            features = train_df.iloc[start_date_idx + 48 * 7:start_date_idx + 48 * 9, :].copy()
            features = features.reset_index(drop=True)

            for i in range(1, num_samples):
                start_date_idx = train_df.index.get_loc(self.searched_result[file_name][i])
                next_features = train_df.iloc[start_date_idx + 48 * 7:start_date_idx + 48 * 9, :].copy()
                next_features = next_features.reset_index(drop=True)
                features = features + next_features

            features = features / num_samples

        elif self.method == 'pattern_median':
            start_date_idx = train_df.index.get_loc(self.searched_result[file_name][num_samples//2])
            features = train_df.iloc[start_date_idx + 48 * 7:start_date_idx + 48 * 9, :].copy()

        return features

    def make_features_pattern(self, num_samples: int = 10, searched_result: dict = None):
        import copy
        train_df = pd.read_csv(self.timestamped_train_dir)
        train_df = train_df.set_index('Timestamp')

        saving_path = f'{self.test_files_dir}/feature_added_{method}'
        p = Path(saving_path)

        if not p.is_dir():
            p.mkdir(parents=True)

        if not searched_result:
            searched_result = self.searched_result

        for file_name in searched_result.keys():
            temp = pd.read_csv(self.test_files_dir + '/timestamped/' + file_name).copy()
            temp['Timestamp'] = pd.to_datetime(temp['Timestamp'])
            target_df = temp.set_index('Timestamp')

            result_df_columns = copy.deepcopy(self.feature_columns)
            result_df_columns.append('TARGET')
            target_df = target_df[result_df_columns]

            features = self.get_features_from_train(train_df, file_name, num_samples=num_samples)
            features = features[result_df_columns]

            target_df = target_df.append(features)
            date_range = pd.date_range(target_df.index[0], periods=48 * 9, freq='30min')
            target_df.index = date_range
            target_df.to_csv(saving_path + f'/{file_name}')

        print(f"all test set csvs are saved under {saving_path}")


if __name__ == '__main__':
    timestamped = "./data/Timestamped.csv"
    test_dir = '../data/valid/timestamped'

    method = ['pattern_mean', 'pattern_median', 'pattern']
    maker = FeatureMaker(test_dir, timestamped, test_end=20, feature_columns=['DHI', 'DNI', 'WS', 'RH', 'T'])
    result = maker.return_search_result()

    for m in method:
        maker.make_features_pattern(num_samples=10, searched_result=result)
