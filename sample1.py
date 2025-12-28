import pandas as pd

# 读取 Excel 文件
df = pd.read_excel("release_data/sup_real_release_data.xlsx")  # 替换为你的文件名

# 显示列名确认 Time 是否为正确列名
print("列名预览:", df.columns)

# 筛选出 Time 不为 0 的数据
df_filtered = df[df['time'] != 0]

# 可选：重置索引
df_filtered = df_filtered.reset_index(drop=True)

# 输出结果预览
print(df_filtered.head())

# 可选：保存筛选结果
df_filtered.to_excel("release_data/time_not_zero.xlsx", index=False)
