import pandas as pd
import glob
import os
import json
import re

# Get a sorted list of all Kickstarter CSV files (assuming they are in the current directory)
csv_files = sorted(glob.glob("Kickstarter*.csv"))
# Read each CSV file into a DataFrame and store them in a list
dfs = [pd.read_csv(file) for file in csv_files]

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Convert 'created_at' and 'deadline' from Unix timestamp to datetime
combined_df['created_at'] = pd.to_datetime(combined_df['created_at'], unit='s')
combined_df['launched_at']   = pd.to_datetime(combined_df['launched_at'], unit='s')
combined_df['deadline']   = pd.to_datetime(combined_df['deadline'], unit='s')

# Find matching project names:
def extract_project_name(url):
    match = re.search(r"projects/[^/]+/([^/]+)/comments", url)
    return match.group(1) if match else None

aggregated_data = {}

json_files = glob.glob("../webscraper/scraped_data_final/*.json")
project_names = set()
for file_path in json_files:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        project_name = extract_project_name(data["url"])
        if project_name:
            project_names.add(project_name)
            aggregated_data[project_name] = data.get("comments", [])

with open("aggregated_comments.json", "w", encoding="utf-8") as outfile:
    json.dump(aggregated_data, outfile, indent=4)

matched_rows = combined_df[combined_df["slug"].isin(project_names)]
matched_rows = matched_rows.sort_values(by=["converted_pledged_amount"], ascending=True)
duplicates_removed = matched_rows.drop_duplicates(subset=["slug"], keep="first")
duplicates_removed["time_passed"] = pd.to_datetime("2025-03-12") - duplicates_removed["created_at"]
relevant_features = ["slug", "percent_funded", "pledged", "goal", "time_passed"] # Pledged and goal are in original currency
final_df = duplicates_removed[relevant_features]

final_df.to_html("output.html")
final_df.to_csv("output.csv")