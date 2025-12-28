import json

with open('supple_1_data_without_protein.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 3. 遍历字典的 *值* 并修改列表
print("正在处理数据...")
# 使用 .values() 来获取所有样本的字典
for sample in data.values():
    if 'time' in sample and 'release_percentage' in sample:
        # 在 time 列表的开头插入 0
        sample['time'].insert(0, 0)

        # 在 release_percentage 列表的开头插入 0
        sample['release_percentage'].insert(0, 0)

print("数据处理完成！")

# 4. (可选) 打印ID为 "49" 的样本来验证结果
print("\n--- 验证修改后的样本 '49' ---")
# 通过键 "49" 来访问特定的样本
print(json.dumps(data['49'], indent=4))
print("------------------------------")

# 5. 将修改后的数据保存到一个新的JSON文件中
output_filename = 'supple_1_data_modified.json'
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"\n修改后的数据已成功保存到文件: {output_filename}")
