import catboost as cb
import json
from datetime import datetime, timedelta
import os
import joblib
import csv


# 加载 XGBoost 模型和scaler
def load_model_and_scaler(model_path, scaler_path):
    model = cb.CatBoostRegressor()
    model.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# 预测函数
def predict_sbi(model, scaler, data, tot):
    data_scaled = scaler.transform([data])
    predicted_sbi = model.predict(data_scaled)[0]
    # 确保预测值不小于0，且不大于tot
    return max(min(predicted_sbi, tot), 0)


# 这里设置您的站点ID列表
station_id_list = []
with open('html.2023.final.data/sno_test_set.txt', 'r') as f:
    for line in f.readlines():
        station_id_list.append(line.strip())


# 主函数
def main(prediction_date_str):
    prediction_date = datetime.strptime(prediction_date_str, "%Y%m%d")

    for station_id in station_id_list:
        if station_id == '500101020':  # 對特定站點直接預測 0
            for hour in range(24):
                for minute in (0, 20, 40):
                    predictions.append(
                        f"{prediction_date_str}_{station_id}_{hour:02d}:{minute:02d},0")
            continue
        model_path = f'catboost_folder/model/{station_id}/catboost_model.json'
        scaler_path = f'catboost_folder/model/{station_id}/scaler.joblib'
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            continue

        bst, scaler = load_model_and_scaler(model_path, scaler_path)

        latest_data_date = prediction_date - timedelta(days=1)
        while True:
            latest_data_path = f'data_transform/{latest_data_date.strftime("%Y%m%d")}/{station_id}.json'
            if os.path.exists(latest_data_path):
                break
            latest_data_date -= timedelta(days=1)

        with open(latest_data_path, 'r', encoding='utf-8') as file:
            latest_station_data = json.load(file)

        latest_time_str = sorted(latest_station_data.keys())[-1]
        latest_data = latest_station_data[latest_time_str]

        for hour in range(24):
            for minute in (0, 20, 40):
                time_str = f"{hour:02d}{minute:02d}"
                act = latest_data.get('act', 0)
                tot = latest_data.get('tot', 0)
                x_input = [
                    int(prediction_date.strftime("%m%d")),
                    prediction_date.weekday(),
                    int(time_str), tot, act
                ]

                predicted_sbi = predict_sbi(bst, scaler, x_input, tot)
                predictions.append(
                    f"{prediction_date_str}_{station_id}_{hour:02d}:{minute:02d},{predicted_sbi}")
    print(f"{prediction_date_str}的预测结果已保存到 {predict_path} 文件中。")


predict_path = 'catboost_folder/predictions_catboost.csv'

if __name__ == "__main__":
    predictions = []
    for day in range(1, 5):
        main(f"2023102{day}")
    for day in range(4, 11):
        main(f"202312{day:02d}")

    with open(predict_path, 'w', newline='') as csvfile:
        prediction_writer = csv.writer(csvfile)
        prediction_writer.writerow(['id', 'sbi'])
        for prediction in predictions:
            prediction_writer.writerow(prediction.split(','))
