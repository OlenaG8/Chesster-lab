import pandas as pd

data = pd.read_parquet('/home/olena/chesster-datasets/train_duck_7/meta/episodes/chunk-000')
print(data.head(1).to_json())
