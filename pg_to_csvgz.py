# convert .pg to .csv.gz

import pandas as pd

parent_path = 'example_data'  # directory path of your data
standardized_data_path = f'{parent_path}/data_standardized/sharable_standardized_v20250723.pq'
standardized_data = pd.read_parquet(standardized_data_path)
standardized_data_csvgz_path = f'{parent_path}/data_standardized/sharable_standardized_v20250723.csv.gz'
standardized_data.to_csv(standardized_data_csvgz_path, index=False, compression='gzip')
