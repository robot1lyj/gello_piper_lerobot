import os
import pyarrow.parquet as pq

recomment_file_path = "/home/lyj/data6/data/chunk-000"
table2 = pq.read_table(recomment_file_path)
target_schema = table2.schema

dir = "/home/lyj/data3/data/chunk-000/"
target_dir = "/home/lyj/data3/data/chunk-001/"

os.makedirs(target_dir, exist_ok=True)

for filename in os.listdir(dir):
    if filename.endswith('.parquet'):
        file_path = os.path.join(dir, filename)
        try:
            table1 = pq.read_table(file_path)
            table1 = table1.cast(target_schema)
            target_file_path = os.path.join(target_dir, filename)
            pq.write_table(table1, target_file_path)
        except Exception as e:
            print(f"rasie an error when processing {file_path}: {e}")
