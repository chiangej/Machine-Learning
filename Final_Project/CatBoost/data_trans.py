import json
import os
from datetime import datetime, timedelta


def aggregate_data(station_data):
    time_points = [f"{hour:02d}:{minute:02d}" for hour in range(24) for minute in range(0, 60, 20)]

    aggregated_data = {}
    for time_point in time_points:
        total_sbi = 0
        count = 0
        first_valid_data = None

        if time_point == "00:00":
            for i in range(3):
                time_to_check = f"00:0{i}"
                if time_to_check in station_data and 'sbi' in station_data[time_to_check]:
                    if first_valid_data is None:
                        first_valid_data = station_data[time_to_check]
                    total_sbi += station_data[time_to_check]['sbi']
                    count += 1
        else:
            current_time = datetime.strptime(time_point, "%H:%M")
            for i in range(3):
                if i == 0:
                    time_to_check = (current_time + timedelta(minutes=i)).strftime("%H:%M")
                    if time_to_check in station_data and 'sbi' in station_data[time_to_check]:
                        if first_valid_data is None:
                            first_valid_data = station_data[time_to_check]
                        total_sbi += station_data[time_to_check]['sbi']
                        count += 1
                else:
                    # deal with negative value
                    time_to_check = (current_time + timedelta(minutes=-i)).strftime("%H:%M")
                    if time_to_check in station_data and 'sbi' in station_data[time_to_check]:
                        if first_valid_data is None:
                            first_valid_data = station_data[time_to_check]
                        total_sbi += station_data[time_to_check]['sbi']
                        count += 1
                    # deal with possitive value
                    time_to_check = (current_time + timedelta(minutes=i)).strftime("%H:%M")
                    if time_to_check in station_data and 'sbi' in station_data[time_to_check]:
                        if first_valid_data is None:
                            first_valid_data = station_data[time_to_check]
                        total_sbi += station_data[time_to_check]['sbi']
                        count += 1

            if count == 0:
                # 擴大搜尋範圍
                for i in range(6):
                    if i == 0:
                        time_to_check = (current_time + timedelta(minutes=i)).strftime("%H:%M")
                        if time_to_check in station_data and 'sbi' in station_data[time_to_check]:
                            if first_valid_data is None:
                                first_valid_data = station_data[time_to_check]
                            total_sbi += station_data[time_to_check]['sbi']
                            count += 1
                    else:
                        # deal with negative value
                        time_to_check = (current_time + timedelta(minutes=-i)).strftime("%H:%M")
                        if time_to_check in station_data and 'sbi' in station_data[time_to_check]:
                            if first_valid_data is None:
                                first_valid_data = station_data[time_to_check]
                            total_sbi += station_data[time_to_check]['sbi']
                            count += 1
                        # deal with possitive value
                        time_to_check = (current_time + timedelta(minutes=i)).strftime("%H:%M")
                        if time_to_check in station_data and 'sbi' in station_data[time_to_check]:
                            if first_valid_data is None:
                                first_valid_data = station_data[time_to_check]
                            total_sbi += station_data[time_to_check]['sbi']
                            count += 1

        if count > 0 and first_valid_data:
            avg_sbi = total_sbi / count
            avg_bemp = first_valid_data.get('tot', 0) - avg_sbi
            aggregated_data[time_point] = {
                'tot': first_valid_data.get('tot', 0),
                'sbi': avg_sbi,
                'bemp': avg_bemp,
                'act': first_valid_data.get('act', '0')
            }

    return aggregated_data


# 路径设置
source_folder = 'html.2023.final.data/release'
target_folder = 'data_transform'
os.makedirs(target_folder, exist_ok=True)

# 这里设置您的站点ID列表
station_id_list = []
with open('html.2023.final.data/sno_test_set.txt', 'r') as f:
    for line in f.readlines():
        station_id_list.append(line.strip())

# 遍历每个日期的文件夹
for date_folder in os.listdir(source_folder):
    date_path = os.path.join(source_folder, date_folder)
    if os.path.isdir(date_path):
        target_date_folder = os.path.join(target_folder, date_folder)
        os.makedirs(target_date_folder, exist_ok=True)

        # 遍历每个站点的文件
        for station_file in os.listdir(date_path):
            if station_file.strip('.json') not in station_id_list:
                continue
            station_path = os.path.join(date_path, station_file)
            with open(station_path, 'r', encoding='utf-8') as file:
                station_data = json.load(file)
                aggregated_data = aggregate_data(station_data)

                # 写入新的 JSON 文件
                target_file_path = os.path.join(target_date_folder, station_file)
                with open(target_file_path, 'w', encoding='utf-8') as target_file:
                    json.dump(aggregated_data, target_file, indent=4)
    print(date_folder)

print("Data transformation complete for all dates.")
