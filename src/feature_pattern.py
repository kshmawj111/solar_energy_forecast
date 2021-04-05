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
                 timestamped_train_path: str,
                 method: str,
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
        self.searched_result = None
        self.test_file_name_list = [f'/{x}.csv' for x in range(test_start, test_end)]

    def return_search_result(self):
        if self.searched_result:
            return self.searched_result

    def cal_best_similar(self) -> dict:
        """
            2021/04/05 comment 추가

            인자로 주어지는 test_start부터 test_end까지에 해당하는 번호를 가진 테스트용 데이터 셋 존재.
            이 테스트 데이터 셋은 후에 모델에서 validation 혹은 prediction 위해 사용되는 데이터 셋임

            위에 언급하였듯이 DeepAR의 경우에는 예측 하려는 기간의 피쳐 값이 존재해야하지 multi-variate하게 예측할 수 있는
            이상한 알고리즘을 가지고 있음.

            이를 실제로 실행하기 위해 해당 피쳐값을 적당히 채워주는 클래스인데 best_similar 메소드는
            총 7일치의 데이터만 있는 테스트 셋과 가장 비슷한 7일치의 데이터를 학습용 데이터에서 찾아낸다.

            즉, 16년 03/04~03/10일까지의 데이터가 테스트 셋에 있다면 해당 기간의 값과 제일 차이가 나지 않는 구간을
            학습 데이터에서 17년 03/15~03/22로 찾아냈다고 하자. 이후 이틀치를 예측하는 것이 목표이므로 17년 03/23~ 03/24까지의
            피쳐 값을 그대로 들고와 16년 03/04~03/10일에 해당하는 데이터 셋으로 채워 넣는 메소드이다.


        """
        result = {}

        for file_num in tqdm(range(self.test_start, self.test_end), desc='Finding the most similar features for test set from training set'):
            test_file_name = f'/{file_num}.csv'
            test_df = pd.read_csv(self.test_files_dir + test_file_name, index_col=0)  # n_th test file 아 n번째 테스트용 파일
            train_df = pd.read_csv(self.timestamped_train_dir)
            search_result = {}

            for day in range(0, 365*3 - 6):
                feature_train = train_df.loc[(day*48):(day+7)*48 -1, ].copy() # 총 7일치 데이터를 학습용 데이터에서 가져옴
                feature_train = feature_train[self.feature_columns] # 필요한 피쳐만 빼옴
                feature_train = feature_train.reset_index(drop=True)

                feature_test = test_df[self.feature_columns]
                feature_test = feature_test.reset_index()

                loss_df = feature_train - feature_test  # 차이 계산

                loss_df = loss_df.apply(lambda row: row.apply(lambda x: x*x)) # Squared Error 구하기
                loss = loss_df.sum().sum()
                search_result[train_df.loc[day*48, 'Timestamp']] = loss # 시작 날짜를 키로 하여 loss 기록

            search_result = pd.DataFrame.from_dict(search_result, orient='index')
            search_result.columns = [test_file_name]
            search_result = search_result[test_file_name].sort_values(ascending=True)
            result[test_file_name[1:]] = search_result.index[1:]

        self.searched_result = result   # searched result는 test셋의 번호를 키로 가지면서
                                        # value로 가장 유사한 시작 날짜를 오름차순 리스트로 가진다.
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

        elif self.method == 'pattern_similar':
            self.cal_best_similar()
            self.make_features()

        return features

    def make_features(self, num_samples: int = 10):
        import copy
        train_df = pd.read_csv(self.timestamped_train_dir)
        train_df = train_df.set_index('Timestamp')

        saving_path = f'{self.test_files_dir}/feature_added_{method}'
        p = Path(saving_path)

        if not p.is_dir():
            p.mkdir(parents=True)

        for file_name in self.test_file_name_list:
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
    timestamped = r"C:\Users\kshma\PycharmProjects\solar_energy_forecast\data\Timestamped.csv"
    test_dir = '../data/valid/timestamped'

    method = ['pattern_mean', 'pattern_median', 'pattern']
    maker = FeatureMaker(test_dir, timestamped, 'pattern_mean', test_end=5, feature_columns=['DHI', 'DNI', 'WS', 'RH', 'T'])
    result = maker.return_search_result()

    for m in method:
        maker.cal_best_similar()
