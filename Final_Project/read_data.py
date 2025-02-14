import panda as pd

# 从 JSON 文件读取数据
df = pd.read_json("C:\Users\chiangej\PycharmProjects\ML_final_project\20231002")

# 将 DataFrame 写入 CSV 文件
df.to_csv('output.csv', index=False)
