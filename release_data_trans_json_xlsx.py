import json
import pandas as pd
import numpy as np

# 读取 JSON 文件
with open("release_data/supple_6m_data.json", "r", encoding="utf-8") as f:
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

    # if sample_id == "18" or sample_id == "28" or sample_id == "59"\
    #         or sample_id == "53" or sample_id == "55" or sample_id == "67":
    #     continue

    if sample_id == "18" or sample_id == "28" or sample_id == "59":
        continue
    # if sample_id not in ["51","52","54","56","59","60","61","68","71","72","93","94"]:
    #     continue
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

    carrier_properties = ["Polymer_MW", "LA/GA", "Polymer_size", "Zeta_Potential", "PDI", "EE"]
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

if "release_percentage" in df.columns and pd.api.types.is_numeric_dtype(df["release_percentage"]):
    df["release_percentage"] = df["release_percentage"] * 0.01
else:
    print("警告：'release_percentage' 列不存在或不是数值类型，无法进行缩放。")

df.to_excel("release_data/sup_6m_release_data_slim.xlsx", index=False)

print("✅ 完成：文件保存为xlsx格式")
