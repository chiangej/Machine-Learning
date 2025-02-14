import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
# import geotable

import os
import pandas as pd
import numpy as np
from math import radians, cos, sin, sqrt, atan2


# def read_data(date, station_id):
#     if date == '20231024':
#         date = '20231031'
#     # read data
#     root_directory = r"C:\Users\chiangej\Desktop\data\html.2023.final.data\release"
#     file_path = os.path.join(root_directory, date, f"{station_id}.json")
#     data = pd.read_json(file_path)
#     data = data.T  # transport the dataframe
#
#     # fill nan
#     df_fill_back = data.fillna(method='backfill', inplace=False)  # fill the empty with the backward data
#     df_fill_front = df_fill_back.fillna(method='ffill', inplace=False)  # fill the last data with the front data
#
#     # date process
#     df_filtered = df_fill_front.iloc[::20]  # time interval = 20 min
#     df_filtered = df_filtered.reset_index(names="datetime")
#     df_filtered['datetime'] = pd.to_datetime(df_filtered['datetime'])  # change type into datetime
#     if(date=="20231031"):
#         date = '20231024'
#     year = int(date[0:4])
#     month = int(date[4:6])
#     day = int(date[6:8])
#     df_filtered['datetime'] = df_filtered['datetime'].apply(lambda x: x.replace(year=year, month=month, day=day))
#     df_filtered = df_filtered.set_axis(["datetime", 'tot', 'sbi', 'bemp', "act"], axis=1)
#     return df_filtered


def feature_process(date_data, df_filtered, station_id):
    # add minute of day
    # df_filtered['minute_of_day'] = df_filtered['datetime'].dt.hour * 60 + df_filtered['datetime'].dt.minute

    # add day of week
    df_filtered["day_of_week"] = df_filtered["datetime"].dt.weekday

    # add station id
    df_filtered["id"] = station_id

    # add longitude latitude
    path = r"C:\Users\chiangej\Desktop\data\html.2023.final.data\demographic.json"
    station_info = pd.read_json(path, convert_axes=False)
    # get data from json
    lat = station_info.loc['lat', station_id]
    lng = station_info.loc['lng', station_id]
    df_filtered["lat"] = lat
    df_filtered["lng"] = lng

    # add temperature rain
    if date_data[4:6] == '10':
        temp = pd.read_csv("202310-Temperature.csv")
        rain = pd.read_csv("C:/Users/chiangej/PycharmProjects/ML_final_project/private_test1/202310-rain.csv")
    elif (date_data[4:6] == '11'):
        temp = pd.read_csv("C:/Users/chiangej/PycharmProjects/ML_final_project/private_test1/202311-Temperature.csv")
        rain = pd.read_csv("C:/Users/chiangej/PycharmProjects/ML_final_project/private_test1/202311-rain.csv")
    else:

        temp = pd.read_csv("C:/Users/chiangej/PycharmProjects/ML_final_project/private_test1/202312-Temperature.csv")
        rain = pd.read_csv("C:/Users/chiangej/PycharmProjects/ML_final_project/private_test1/202312-rain.csv")

    date = date_data[-2:]
    # 提取名为 column_name 的列
    temp_data = list(temp.loc[int(date) - 1])
    rain_data = list(rain.loc[int(date) - 1])

    repeat = []
    repeat_rain = []
    for i in range(1, 25):
        if temp_data[i] == '--':
            temp_data[i] = 0
        if rain_data[i] == '--':
            rain_data[i] = 0
        repeat.extend(np.tile(float(temp_data[i]), 3))
        repeat_rain.extend(np.tile(float(rain_data[i]), 3))
    expanded_data = pd.DataFrame(repeat, columns=['temp'])
    df_filtered = pd.concat([df_filtered, expanded_data], axis=1)
    expanded_data = pd.DataFrame(repeat_rain, columns=['rain'])
    df_filtered = pd.concat([df_filtered, expanded_data], axis=1)

    df_filtered.drop('datetime', axis=1, inplace=True)
    df_filtered.drop('bemp', axis=1, inplace=True)

    # add distance
    # the coordinate of nearby MRT station
    station_cord = [(25.023777, 121.553115), (25.026125, 121.543437), (25.032943, 121.543551), (25.041629, 121.543767),
                    (25.033102, 121.563292), (25.033326, 121.553526), (25.032943, 121.543551), (25.033396, 121.534882),
                    (25.033847, 121.528739), (25.032729, 121.51827), (25.041256, 121.51604), (25.046255, 121.517532),
                    (24.9921276, 121.5406037), (25.001853, 121.539051), (25.014908, 121.534216),
                    (25.020725, 121.528168), (25.026357, 121.522873), (25.032729, 121.51827), (25.013821, 121.515485),
                    (25.042356, 121.532905), (25.052015, 121.533075), (25.018535, 121.558791), (25.041629, 121.543767),
                    (25.052319, 121.544011), (25.03283, 121.569576), (25.041256, 121.51604), (24.975169, 121.542942),
                    (25.051836, 121.55153)
                    ]
    distance = haversine((lat, lng), station_cord)
    df_filtered["distance"] = distance

    return df_filtered


def haversine(coord1, coord_list_2):
    distance = float("inf")
    for coord2 in coord_list_2:
        # 将经纬度转换为弧度
        lat1, lon1 = map(radians, coord1)
        lat2, lon2 = map(radians, coord2)

        # Haversine 公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # 地球半径 (公里)
        r = 6371

        if c * r < distance:
            distance = c * r
    # 结果 (公里)
    return distance * 1000


def read_data(date_folder, station_id):

    # if(date_folder == "20231024"):
    #     date_folder = '20231030'
    # 讀取資料
    root_directory = r"C:\Users\chiangej\Desktop\data\html.2023.final.data\release"
    file_path = os.path.join(root_directory, date_folder, f"{station_id}.json")
    data = pd.read_json(file_path)
    csv_file_path = station_id + '.csv'
    data.to_csv(csv_file_path, index=False)
    train = pd.read_csv(csv_file_path)
    df = train.T
    df = df.fillna(method='backfill', inplace=False)
    df = df.fillna(method='ffill', inplace=False)
    # df = nan(df)
    # 每20分鐘取一個點
    df_filtered = df.iloc[::20]
    df_filtered = df_filtered.reset_index(names="datetime")
    df_filtered['datetime'] = pd.to_datetime(df_filtered['datetime'])
    df_filtered['minute_of_day'] = df_filtered['datetime'].dt.hour * 60 + df_filtered['datetime'].dt.minute
    date = datetime.strptime(date_folder, '%Y%m%d')
    # df_filtered["dayofweek"] = date.weekday()
    df_filtered.drop('datetime', axis=1, inplace=True)
    df_filtered = df_filtered.set_axis(['tot', 'sbi', 'bemp', "act", "min"], axis=1)
    df_filtered.drop('tot', axis=1, inplace=True)

    if (date_folder[4:6] == '10'):
        temp = pd.read_csv("202310-Temperature.csv")
        rain = pd.read_csv("C:/Users/chiangej/PycharmProjects/ML_final_project/private_test1/202310-rain.csv")

    elif (date_folder[4:6] == '11'):
        temp = pd.read_csv("C:/Users/chiangej/PycharmProjects/ML_final_project/private_test1/202311-Temperature.csv")
        rain = pd.read_csv("C:/Users/chiangej/PycharmProjects/ML_final_project/private_test1/202311-rain.csv")

    else:
        temp = pd.read_csv("C:/Users/chiangej/PycharmProjects/ML_final_project/private_test1/202312-Temperature.csv")
        rain = pd.read_csv("C:/Users/chiangej/PycharmProjects/ML_final_project/private_test1/202312-rain.csv")

    date = date_folder[-2:]
    # 提取名为 column_name 的列
    temp_data = list(temp.loc[int(date) - 1])  # 放入日期-1
    rain_data = list(rain.loc[int(date) - 1])  # 放入日期-1

    repeat = []
    repeat_rain = []
    for i in range(1, 25):
        if (temp_data[i] == '--'):
            temp_data[i] = 5
        if (rain_data[i] == '--'):
            rain_data[i] = 0
        repeat.extend(np.tile(float(temp_data[i]), 3))
        repeat_rain.extend(np.tile(float(rain_data[i]), 3))
    expanded_data = pd.DataFrame(repeat, columns=['temp'])
    df_filtered = pd.concat([df_filtered, expanded_data], axis=1)
    expanded_data = pd.DataFrame(repeat_rain, columns=['rain'])
    df_filtered = pd.concat([df_filtered, expanded_data], axis=1)
    # expanded_data = pd.DataFrame(repeat, columns=['holiday'])
    # df_filtered = pd.concat([df_filtered, expanded_data], axis=1)
    # df_filtered['holiday'] = 0
    # if date_folder == '20231009':
    #     df_filtered['holiday'] = 1
    return df_filtered





def nan(train):
    train = train.T
    for index, row in train.iterrows():
        for col in range(len(train.columns)):
            # 檢查是否為NaN
            if pd.isna(row[col]):
                # 獲取左邊和右邊的值
                left = row[col - 1] if col > 0 else np.nan
                right = row[col + 1] if col < len(train.columns) - 1 else np.nan
                # 計算平均值，忽略NaN
                avg = np.nanmean([left, right])
                # 填充NaN值
                train.iat[index, col] = avg
    return train.T


def normalize(df):
    columns_to_normalize = ['sbi', "min"]
    for column in columns_to_normalize:
        min_val = df[column].min()
        max_val = df[column].max()
        if min_val != max_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df

def normalize2(df):
    columns_to_normalize = ["sbi", "tot","lng","lat","rain","distance","minute_of_day","id"]
    for column in columns_to_normalize:
        min_val = df[column].min()
        max_val = df[column].max()
        if min_val != max_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df


def build_trainx(df):
    df = df.drop('sbi', axis=1, inplace=False)
    return df


def build_trainy(df):
    return np.array(df["sbi"])


def denormalize(df, max, min):
    if max != min:
        df["sbi"] = (df["sbi"] * (max - min) + min)
    return df


def id_list(day, mc):
    start_date = datetime(2023, 10, day)
    end_date = datetime(2023, 10, day, 23, 40)
    interval = timedelta(minutes=20)
    middle_code = mc
    current_date = start_date
    date_list = []
    while current_date <= end_date:
        formatted_date = current_date.strftime("%Y%m%d_%H:%M")
        formatted_datetime = f"202310{day}_{middle_code}_{formatted_date.split('_')[-1]}"
        date_list.append(formatted_datetime)
        current_date += interval
    return date_list
