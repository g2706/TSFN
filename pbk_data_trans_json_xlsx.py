import json
import pandas as pd
import numpy as np

# 读取 JSON 文件
with open("PBK_data/pbk_1_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for sample_id, sample in data.items():
    # 基本信息
    drug_name = sample.get("drug_name", "")
    source = sample.get("source", [""])[0]
    features = sample.get("feature", [])
    drug_features = sample.get("drug", [])
    times = sample.get("time", [])
    releases = sample.get("release_percentage", [])

    # 处理 feature：转为 float，非法填 NaN
    features = [
        float(f) if isinstance(f, (int, float)) or
                    (isinstance(f, str) and f.replace('.', '', 1).replace('-', '', 1).isdigit())
        else np.nan for f in features
    ]

    # 处理 drug 特征
    processed_drug = []
    for i, d in enumerate(drug_features):
        if i >= 7 and i <= 10:  # 布尔型字段
            if isinstance(d, str):
                d = d.lower().strip()
                if d == "true":
                    processed_drug.append(1)
                elif d == "false":
                    processed_drug.append(0)
                else:
                    processed_drug.append(np.nan)
            elif isinstance(d, bool):
                processed_drug.append(int(d))
            else:
                processed_drug.append(np.nan)
        else:  # 普通数值字段
            if isinstance(d, (int, float)):
                processed_drug.append(float(d))
            elif isinstance(d, str) and d.replace('.', '', 1).replace('-', '', 1).isdigit():
                processed_drug.append(float(d))
            else:
                processed_drug.append(np.nan)

    carrier_properties = ["Weight", "Dose", "End", "LA/GA", "Polymer_MW", "glass_temperature", "Drug_loading", "Polymer_size", "PDI", "EE"]
    drug_properties = ["Drug_MW", "LogP", "HBD", "HBA", "HA", "RB", "PSA", "LR_MW", "LR_LogP", "LR_HBD", "LR_HBA", "EMW", "NAR", "FSP3"]

    # 每个时间点展开为一行
    for t, r in zip(times, releases):
        row = {
            "sample_id": sample_id,
            "source": source,
            "drug_name": drug_name,

            **{carrier_properties[i]: val for i, val in enumerate(features)},
            **{drug_properties[i]: val for i, val in enumerate(processed_drug)},
            "time": t,
            "release_percentage": r
        }
        rows.append(row)

# 写入 Excel
df = pd.DataFrame(rows)

# 用每一列的均值填充缺失值（仅限数值列）
df = df.apply(lambda col: col.fillna(col.mean()) if col.dtype in [np.float64, np.int64] else col)

# 确保 'release_percentage' 列存在且为数值类型
if "release_percentage" in df.columns and pd.api.types.is_numeric_dtype(df["release_percentage"]):
    print("正在根据每个 sample_id 组内的最大值进行百分比化...")

    # 【核心修改】按 sample_id 分组，然后用组内每个值除以该组的最大值，再乘以 100
    # 使用 transform 来确保结果可以直接赋值回原 DataFrame
    df['release_percentage'] = df.groupby('sample_id')['release_percentage'].transform(
        lambda x: (x / x.max()) * 1 if x.max() != 0 else 0
    )

else:
    print("警告：'release_percentage' 列不存在或不是数值类型，无法进行百分比化。")

df.to_excel("PBK_data/pbk_sup_1_data.xlsx", index=False)

print("✅ 完成：文件保存为xlsx格式")
