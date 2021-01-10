import pandas as pd
from itertools import accumulate
from pathlib import Path


class Timestamper:
    def __init__(self, test_path): # test_path must navigate to the root folder containing test csvs
        self.__test_path = test_path
        self.__test_timestamped_path = test_path + '/timestamped'


    def stamp(self):

        for file_num in range(0, 81):
            filename = f'/{file_num}.csv'
            test_file_path = self.__test_path + filename

            if not Path(self.__test_timestamped_path).is_dir():
                Path(self.__test_timestamped_path).mkdir()

            # only do the job if file doesn't exist
            if not Path(self.__test_timestamped_path+filename).is_file():
                df = pd.read_csv(test_file_path, index_col=0)
                df = df.reset_index()

                df['Year'] = df['Day'] / 365 + 2021  # Just an assumption
                df['Year'] = df['Year'].apply(lambda x: int(x))
                df['Month'] = df['Day'].apply(self.find_month)
                df = self.set_days(df)

                df['Timestamp'] = df.apply(
                    lambda row: self.to_str(row['Year'], row['Month'], row['Day'], row['Hour'], row['Minute']), axis=1)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])

                df = df.sort_values('Timestamp')
                df = df.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1)
                save_path = self.__test_timestamped_path + f'/{file_num}.csv'
                df.to_csv(save_path, index=False)

        return self.__test_timestamped_path

    def to_str(self, y, mo, d, h, m):
        mo = str(int(mo)).zfill(2)
        d = str(int(d)).zfill(2)
        h = str(int(h)).zfill(2)
        m = str(int(m)).zfill(2)
        time_string = f'{int(y)}-{mo}-{d} {h}:{m}:00'
        return time_string

    def find_month(self, day):
        month_table = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month_table = list(accumulate(map(int, month_table)))
        # [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
        day = (day % 365) + 1
        if day <= 181:  # 6까지
            idx = 0

            while idx < 5:
                if day > month_table[idx]:
                    idx += 1

                else:
                    break

            return idx + 1

        else:
            idx = 6

            while idx < 11:
                if day > month_table[idx]:
                    idx += 1

                else:
                    break

            return idx + 1

    def set_days(self, df):
        month_table = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # 보편적인 1년의 일 수 그냥 넣음
        month_table = list(accumulate(map(int, month_table)))  # 누적하기
        copied = df.copy()
        copied['Day'] = (copied['Day'] % 365) + 1
        modifed = []

        for i, v in enumerate(month_table):
            if i > 0:
                temp = copied[copied['Month'] == i].copy()
                temp['Day'] = temp['Day'] - month_table[i - 1]
                modifed.append(temp)

        return pd.concat(modifed)