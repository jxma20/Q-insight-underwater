import pandas as pd

# 读取CSV文件
df = pd.read_csv('/data2/wyl/work/Q-Align/playground/data/UWIQA/data.csv')

# 在第一列的值后面加上.png
df.iloc[:, 0] = df.iloc[:, 0].astype(str) + '.png'

# 保存修改后的CSV文件
df.to_csv('/data2/wyl/work/Q-Align/playground/data/UWIQA/modified_file.csv', index=False)