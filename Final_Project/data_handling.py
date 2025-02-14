# Excel檔合併
import os
import pandas as pd

station = []
data_frame_final = pd.DataFrame(columns=["id", "sbi"])
with open(r"C:\Users\chiangej\Desktop\data\html.2023.final.data\sno_test_set.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    station.append(lines)
station = [item.strip() for sublist in station for item in sublist]
mon = []
tue = []
sat = []
sun = []

for station_id in station:
    i = 0

    data_folder_path = r"C:\Users\chiangej\PycharmProjects\ML_final_project\public_test_3"
    date_folder = station_id + "_public"
    file_path = os.path.join(data_folder_path, f"{date_folder}.csv")
    data = pd.read_csv(file_path)

    value = data['sbi'].iloc[0:72]
    sat.append(value)

    value2 = data['sbi'].iloc[72:144]
    sun.append(value2)

    value3 = data['sbi'].iloc[144:216]
    mon.append(value3)

    value4 = data['sbi'].iloc[216:288]
    tue.append(value4)

filee = pd.concat(sat, ignore_index=True)
out = filee.to_csv("public_test3_sat.csv", index=False)

filee = pd.concat(sun, ignore_index=True)
out2 = filee.to_csv("public_test3_sun.csv", index=False)

filee = pd.concat(mon, ignore_index=True)
out3 = filee.to_csv("public_test3_mon.csv", index=False)

filee = pd.concat(tue, ignore_index=True)
out4 = filee.to_csv("public_test3_tue.csv", index=False)


