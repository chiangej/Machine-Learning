import catboost as cb
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
from datetime import datetime, timedelta
import os
import joblib


# 自定義數據處理函數
def process_specific_station_data(demographic_path, data_path_template, start_date, end_date,
                                  station_id_to_read):
    with open(demographic_path, 'r', encoding='utf-8') as file:
        station_infos = json.load(file)

    all_x, all_y = [], []
    if station_id_to_read in station_infos:
        date_range = [
            start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)
        ]
        for date in date_range:
            date_str = date.strftime("%Y%m%d")
            MMDD = date.strftime("%m%d")
            file_path = data_path_template.format(date_str, station_id_to_read)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    # print('day:', file_path)
                    station_data = json.load(file)
                    for time, data in station_data.items():
                        if data:
                            time_str = time.replace(':', '')
                            all_x.append([
                                int(MMDD),
                                date.weekday(),
                                int(time_str),
                                data.get('tot', 0),
                                int(data.get('act', 0))
                            ])
                            all_y.append(float(data.get('sbi', 0)))
    # print("Data processing for station {} complete".format(station_id_to_read))
    return np.array(all_x), np.array(all_y)


# 主函數
if __name__ == "__main__":
    demographic_path = 'data_transform/new_demographic.json'
    data_path_template = 'data_transform/{}/{}.json'
    start_date = datetime(2023, 10, 2)
    end_date = datetime(2023, 12, 3)

    # 这里设置您的站点ID列表
    station_id_list = []
    with open('html.2023.final.data/sno_test_set.txt', 'r') as f:
        for line in f.readlines():
            station_id_list.append(line.strip())

    # 加載站點信息
    with open(demographic_path, 'r', encoding='utf-8') as file:
        station_infos = json.load(file)

    station_id_all = []
    station_ids = station_infos.keys()
    for i in station_ids:
        if i in station_id_list:
            station_id_all.append(i)

    total_stations = len(station_id_all) - 1
    processed_stations = 0
    # 預處理所有站點的數據並訓練模型
    for station_id in station_id_all:
        if station_id == '500101020':  # 跳過特定站點 這一站壞了
            continue
        all_x, all_y = process_specific_station_data(demographic_path, data_path_template,
                                                     start_date, end_date, station_id)

        # 檢查數據是否存在
        if len(all_x) == 0 or len(all_y) == 0:
            print(f"站點 {station_id} 沒有足夠數據，跳過訓練。")
            continue

        # 特徵縮放
        scaler = StandardScaler()
        all_x_scaled = scaler.fit_transform(all_x)

        model = cb.CatBoostRegressor(loss_function='RMSE', silent=True, verbose=False)

        # 使用 grid_search 方法尋找最佳參數
        grid = {
            'iterations': [1000, 1050, 1100],
            'learning_rate': [0.12, 0.13, 0.14],
            'depth': [6],
            'l2_leaf_reg': [1],
            'border_count': [51, 52, 53]
        }

        grid_search_result = model.grid_search(grid,
                                               X=all_x_scaled,
                                               y=all_y,
                                               cv=5,
                                               plot=False,
                                               search_by_train_test_split=True,
                                               logging_level='Silent')

        best_model_params = grid_search_result['params']
        print('best parameter:', best_model_params)
        # # 使用最佳參數再次訓練模型
        model = cb.CatBoostRegressor(**best_model_params, loss_function='RMSE', verbose=False)
        model.fit(all_x_scaled, all_y)

        # 保存模型和縮放器
        save_dir = os.path.join('catboost_folder/model', station_id)
        os.makedirs(save_dir, exist_ok=True)
        model.save_model(os.path.join(save_dir, 'catboost_model.json'))
        joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))

        processed_stations += 1
        print(f"站點 {station_id} 處理完成。進度：{processed_stations}/{total_stations} 站點")

    print('over')
