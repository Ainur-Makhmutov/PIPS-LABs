import pandas as pd
import json

# Читаем JSON lines файл
data = []
with open(r'Data\instance_usage-000000000000.json', 'r') as f:
    i = 1
    for line in f:
        line = line.strip()
        i+= 1
        if (i == 288002): break

        if line:
            data.append(json.loads(line))

# Создаем DataFrame
df = pd.DataFrame(data)

# Заполняем NaN пустыми строками (опционально)
df = df.fillna('')

# Сохраняем в CSV
df.to_csv(r'Data\Dataset-24.csv', index=False, encoding='utf-8')

print(f"Сохранено {len(df)} записей")
print(f"Колонки: {list(df.columns)}")