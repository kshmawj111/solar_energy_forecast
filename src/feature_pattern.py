import pandas as pd
from tqdm import tqdm
from pathlib import Path
import copy
import numpy as np

"""
            2021/04/05 comment 추가 (리팩토링 중 추가함)

            인자로 주어지는 test_start부터 test_end까지에 해당하는 번호를 가진 테스트용 데이터 셋 존재.
            이 테스트 데이터 셋은 후에 모델에서 validation 혹은 prediction 위해 사용되는 데이터 셋임

            위에 언급하였듯이 DeepAR의 경우에는 예측 하려는 기간의 피쳐 값이 존재해야하지 multi-variate하게 예측할 수 있는
            이상한 알고리즘을 가지고 있음.

            이를 실제로 실행하기 위해 해당 피쳐값을 적당히 채워주는 클래스인데  cal_best_similar 메소드는
            총 7일치의 데이터만 있는 테스트 셋과 가장 비슷한 7일치의 데이터를 학습용 데이터에서 찾아낸다.

            구체적으로 16년 03/04~03/10일까지의 데이터가 테스트 셋에 있다면 해당 기간의 값과 제일 차이가 나지 않는 구간을
            학습 데이터에서 17년 03/15~03/22로 찾았냈다고하자. 그러면 class 변수 searcehd_result에 key를 해당
            테스트 셋의 번호로, value를 train dataset에서 해당 셋과 가장 오차가 작은 순서대로 정렬하여 가진다.
            
"""
class FeatureMaker:
    def __init__(self,
                 test_path: str,
                 timestamped_train_dir: str,
                 method: str,
                 feature_columns: list,
                 test_file_end: int,
                 test_file_start: int = 0,
                 ):
        self.test_files_dir = test_path
        self.timestamped_train_dir = timestamped_train_dir
        self.train_df = pd.read_csv(timestamped_train_dir)
        self.train_df = self.train_df.set_index('Timestamp')
        self.feature_columns = feature_columns
        self.test_file_start = test_file_start
        self.test_file_end = test_file_end
        self.method = method
        self.searched_result = None
        self.test_file_name_list = [f'{x}.csv' for x in range(test_file_start, test_file_end)]

        
    def return_search_result(self):
        if self.searched_result:
            return self.searched_result

    """
    
    Training 셋에서 피쳐를 생성
    
    """
    def make_features_from_train(self, method: str, saving_path: str, num_samples: int = 10):
        for test_file_name in self.test_file_name_list:
            temp = pd.read_csv(self.test_files_dir + "\\timestamped\\" + test_file_name).copy()
            temp['Timestamp'] = pd.to_datetime(temp['Timestamp'])
            target_df = temp.set_index('Timestamp')

            result_df_columns = copy.deepcopy(self.feature_columns)
            result_df_columns.append('TARGET')
            target_df = target_df[result_df_columns]

            features = self.get_features_from_train(method, test_file_name, num_samples=num_samples)
            features = features[result_df_columns]

            target_df = target_df.append(features)
            date_range = pd.date_range(target_df.index[0], periods=48 * 9, freq='30min')
            target_df.index = date_range
            target_df.to_csv(saving_path + f'\\{test_file_name}')

        print(f"all test set csvs are saved under {saving_path}")
        
        
    def cal_best_similar(self):
        result = {}

        for file_num in tqdm(range(self.test_file_start, self.test_file_end), desc='Finding the most similar features for test set from training set'):
            test_file_name = f'/{file_num}.csv'
            test_df = pd.read_csv(self.test_files_dir + test_file_name, index_col=0)  # n_th test file 아 n번째 테스트용 파일
            search_result = {}

            # TODO: Thread로 병렬화
            for day in range(0, 365*3 - 6):
                feature_train = self.train_df.iloc[(day * 48):(day + 7) * 48, ].copy() # 총 7일치 데이터를 학습용 데이터에서 가져옴
                feature_train = feature_train[self.feature_columns] # 필요한 피쳐만 빼옴
                feature_train = feature_train.reset_index(drop=True)

                feature_test = test_df[self.feature_columns]
                feature_test.reset_index(drop=True, inplace=True) # Day index를 제거

                error = (feature_train - feature_test).to_numpy()  # 차이행렬 구하기
                rsse = np.sqrt(sum(np.sum(error**2, axis=0)))  # root of sum of squared error for each time period
                search_result[self.train_df.index[day*48]] = rsse # 시작 날짜를 키로 하여 error 기록

            search_result = dict(sorted(search_result.items(), key=lambda x: x[1])) # 기록된 오차로 정렬

            # value가 0인 항목의 키는 제거하도록
            for k, v in search_result.items():
                if v == 0:
                    del search_result[k]
                    break

            result[test_file_name[1:]] = list(search_result.keys()) #key는 /를 제거하여 순수 번호만 넣기

        return result   # result는 test셋의 번호를 키로 가지면서 value로 가장 유사한 시작 날짜를 오름차순 리스트로 가진다.


    def get_features_from_train(self, method: str, file_name: str, num_samples: int):
        features = None

        if method == 'first':
            start_date_idx = self.train_df.index.get_loc(self.searched_result[file_name][0])
            features = self.train_df.iloc[start_date_idx + 48 * 7:start_date_idx + 48 * 9, :]

        elif method == 'mean':
            start_date_idx = self.train_df.index.get_loc(self.searched_result[file_name][0])
            features = self.train_df.iloc[start_date_idx + 48 * 7:start_date_idx + 48 * 9, :].copy()
            features = features.reset_index(drop=True)

            for i in range(1, num_samples):
                start_date_idx = self.train_df.index.get_loc(self.searched_result[file_name][i])
                next_features = self.train_df.iloc[start_date_idx + 48 * 7:start_date_idx + 48 * 9, :].copy()
                next_features = next_features.reset_index(drop=True)
                features = features + next_features

            features = features / num_samples

        elif method == 'median':
            start_date_idx = self.train_df.index.get_loc(self.searched_result[file_name][num_samples//2])
            features = self.train_df.iloc[start_date_idx + 48 * 7:start_date_idx + 48 * 9, :].copy()

        return features

    """
        오직 test 셋으로 EMW, EWM 등을 사용하여 추출
    
    """
    
    
    def make_features_from_test(self, method: str, saving_path: str) -> pd.DataFrame:
        test_df = []
        parts = []

        # sort by hour and minute
        sorted_df = self.train_df.sort_values(['Hour', 'Minute'], ascending=True)

        for df_num in range(0, 48):
            part_of_df = sorted_df.iloc[df_num * 7:(df_num + 1) * 7, ]
            parts.append(self.fill_in_prediction_features(part_of_df, method=method))

        test_df.append(pd.concat(parts))
        result = pd.concat(test_df)
        result = result.sort_values(['Day', 'Hour', 'Minute'], ascending=True)
        return result


    # short forecast for features in the range to predict
    def fill_in_prediction_features(self, df: pd.DataFrame, method: str, windows_size: int = 2,
                                    prediction_period: int = 2) -> pd.DataFrame:
        temp = df.copy(deep=True)

        for day in range(prediction_period):
            ma = None

            if method == 'rolling':
                ma = temp.rolling(windows_size).mean()

            elif method == 'ewm':
                ma = temp.ewm(alpha=0.3).mean()

            elif method == 'median':
                # copy the median value to two features
                ma = {}
                columns = temp.columns.copy()

                for col in columns:
                    if col != 'Timestamp':
                        ma[col] = [temp[col].median()]

                ma = pd.DataFrame.from_dict(ma)

            elif method == 'mean':  # 8th is mean of previous 3 days, 9 th is mean of previous 6 days
                columns = temp.columns.copy()
                columns = columns.drop('Timestamp')
                mean_temp = temp[columns]
                ma = mean_temp.iloc[:-(day * 3 + 3), ].mean()

            ma['Day'] = day + 7 + 1
            ma['TARGET'] = 0

            if isinstance(ma, pd.DataFrame):
                temp = temp.append(ma.iloc[-1, :])

            elif isinstance(ma, pd.Series):
                temp = temp.append(ma, ignore_index=True)

        return temp

    """
        Public한 method로 이걸 호출하면 됨
    """

    def make_features(self, method: str, from_train: bool, num_samples: int = 10):
        saving_path = fr'{self.test_files_dir}\feature_added_{method}'
        p = Path(saving_path)

        if not p.is_dir():
            p.mkdir(parents=True)

        if from_train:
            if not self.searched_result:
                self.searched_result = self.cal_best_similar()

            self.make_features_from_train(method, saving_path, num_samples)


        else:
            self.make_features_from_test(method)





if __name__ == '__main__':
    timestamped = r"C:\Users\kshma\PycharmProjects\solar_energy_forecast\data\Timestamped.csv"
    test_dir = r'C:\Users\kshma\PycharmProjects\solar_energy_forecast\data\valid'

    method = ['pattern_mean', 'pattern_median', 'pattern']
    maker = FeatureMaker(test_dir, timestamped, 'pattern_mean', test_file_end=2, feature_columns=['DHI', 'DNI', 'WS', 'RH', 'T'])
    maker.make_features_from_train('pattern')

