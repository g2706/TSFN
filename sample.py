import json

# 读取 JSON 文件（或你可以直接把 json_data 替换为你实际的数据变量）
with open("release_data/supple_6m_data.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

slim=["18","28","59"]
pure=["18","28","59","53","55","67"]

# 用于去重的集合（基于小写）
seen = set()
# 最终药物名列表（格式统一为首字母大写）
drug_list = []

for sample in json_data.values():
    if "drug_name" in sample and isinstance(sample["drug_name"], str):
        raw_name = sample["drug_name"].strip()
        name_lower = raw_name.lower()
        if name_lower not in seen:
            seen.add(name_lower)
            drug_list.append(raw_name.title())  # 首字母大写

# 排序后打印
drug_list.sort()
print(f"共发现 {len(drug_list)} 种药物（已统一首字母大写）：")
for name in drug_list:
    print(name)

