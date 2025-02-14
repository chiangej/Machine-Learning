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
wed = []
thur = []
fri = []
sat = []
sun = []

for station_id in station:
    i = 0

    data_folder_path = r"C:\Users\chiangej\PycharmProjects\ML_final_project\private_test3"
    date_folder = station_id + "_private"
    file_path = os.path.join(data_folder_path, f"{date_folder}.csv")
    data = pd.read_csv(file_path)

    value = data['sbi'].iloc[0:72]
    mon.append(value)

    value = data['sbi'].iloc[72:144]
    tue.append(value)

    value = data['sbi'].iloc[144:216]
    wed.append(value)

    value = data['sbi'].iloc[216:288]
    thur.append(value)

    value = data['sbi'].iloc[288:360]
    fri.append(value)

    value = data['sbi'].iloc[360:432]
    sat.append(value)

    value = data['sbi'].iloc[432:504]
    sun.append(value)



filee = pd.concat(mon, ignore_index=True)
out = filee.to_csv("private_test2_mon.csv", index=False)

filee = pd.concat(tue, ignore_index=True)
out2 = filee.to_csv("private_test2_tue.csv", index=False)

filee = pd.concat(wed, ignore_index=True)
out3 = filee.to_csv("private_test2_wed.csv", index=False)

filee = pd.concat(thur, ignore_index=True)
out4 = filee.to_csv("private_test2_thur.csv", index=False)

filee = pd.concat(fri, ignore_index=True)
out5 = filee.to_csv("private_test2_fri.csv", index=False)

filee = pd.concat(sat, ignore_index=True)
out6 = filee.to_csv("private_test2_sat.csv", index=False)

filee = pd.concat(sun, ignore_index=True)
out7 = filee.to_csv("private_test2_sun.csv", index=False)




