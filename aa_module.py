import numpy as np
import pandas as pd


def find_200():
    df = pd.read_csv('finace_info.csv', header=0)

    df[df['액면가'] == '무액면'] = 0
    df[df['액면가'] == '-'] = 0
    df[['발행주식수', '액면가']] = df[['발행주식수', '액면가']].replace('[\,]', '', regex=True).astype(np.float64)
    df['시가총액'] = df['발행주식수'] * df['액면가']
    df = df.sort_values(['시가총액'], ascending=False)

    top_200 = df['종목코드'][:200]
    top_200.to_csv("a.csv", index=False, encoding='utf-8-sig')
    top_200 = top_200.to_list()

    final = ''
    for x in top_200:
        final += str(x)
    return final

if __name__ == '__main__':
    print(find_200())