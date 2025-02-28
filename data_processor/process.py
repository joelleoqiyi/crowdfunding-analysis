import pandas as pd
import glob

# Get a sorted list of all Kickstarter CSV files (assuming they are in the current directory)
csv_files = sorted(glob.glob("Kickstarter*.csv"))

# Read each CSV file into a DataFrame and store them in a list
dfs = [pd.read_csv(file) for file in csv_files]

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Convert 'created_at' and 'deadline' from Unix timestamp to datetime
combined_df['created_at'] = pd.to_datetime(combined_df['created_at'], unit='s')
combined_df['deadline']   = pd.to_datetime(combined_df['deadline'], unit='s')

