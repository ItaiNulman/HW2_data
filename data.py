import pandas as pd
from datetime import datetime


def load_data(path):
    return pd.read_csv(path)


def add_new_columns(df):
    df['season_name'] = df['season'].apply(
        lambda x: 'Spring' if x == 0 else 'Summer' if x == 1 else 'Fall' if x == 2 else 'Winter')
    # TODO: check if integer
    df['Hour'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M').hour)
    df['Day'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M').day)
    df['Month'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M').month)
    df['Year'] = df['timestamp'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y %H:%M').year)

    df['is_weekend_holiday'] = df['is_weekend'] + 2 * df['is_holiday']  # TODO: check if ok without apply

    df['t_diff'] = df['t2'] - df['t1']

    return df


def data_analysis(df):
    print("describe output:")
    print(df.describe().to_string())
    print()
    print("corr output:")
    corr = df.corr()
    print(corr.to_string())
    print()

    # create dictionary - key are tuple of 2 columns, value is the correlation
    corr_dict = {}
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            corr_dict[(corr.index[i], corr.columns[j])] = abs(corr.iloc[i, j])

    # sort the dictionary by value
    sorted_corr_dict = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)

    # print the 5 most correlated columns

    print("Highest correlated are: ")
    for i in range(5):
        print(f"{i+1}.", sorted_corr_dict[i][0], "with", "{:.6f}".format(sorted_corr_dict[i][1]))
    print()

    # print the 5 least correlated columns

    print("Lowest correlated are: ")
    for i in range(5):
        print(f"{i+1}.", sorted_corr_dict[-i - 1][0], "with", "{:.6f}".format(sorted_corr_dict[-i - 1][1]))
    print()

    # print the mean of t_diff for each season - 'season' average of t_diff is <season_avg_t_diff>
    season_avg_t_diff = df.groupby('season_name')['t_diff'].mean()
    print("fall average t_diff is", "{:.2f}".format(season_avg_t_diff['Fall']))
    print("spring average t_diff is", "{:.2f}".format(season_avg_t_diff['Spring']))
    print("summer average t_diff is", "{:.2f}".format(season_avg_t_diff['Summer']))
    print("winter average t_diff is", "{:.2f}".format(season_avg_t_diff['Winter']))
    print("All average t_diff is", "{:.2f}".format(df['t_diff'].mean()))
    print()

    return
