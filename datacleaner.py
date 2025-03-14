import pandas as pd
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# insert directory where all the Kickstarter datasets are stored
dataset_dir = ""

def clean_kickstarter_data(file_path):
    """Load and clean a single Kickstarter dataset."""
    try:
        print(f"Processing file: {file_path}")  # Real-time update
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path, low_memory=False)
        
        # Convert 'launched_at' to datetime format
        if 'launched_at' in df.columns:
            df['launched_at'] = pd.to_datetime(df['launched_at'], unit='s', errors='coerce')
        
        # Extract 'category_id' and 'category_name' from 'category' column
        if 'category' in df.columns:
            try:
                df['category'] = df['category'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                df['category_id'] = df['category'].apply(lambda x: x.get('id', None) if isinstance(x, dict) else None)
                df['category_name'] = df['category'].apply(lambda x: x.get('name', None) if isinstance(x, dict) else None)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"JSON parsing error in {file_path}: {e}")
                df['category_id'] = None
                df['category_name'] = None
        
        # Use the 'blurb' column as the campaign summary
        if 'blurb' in df.columns:
            df['campaign_summary'] = df['blurb']
        else:
            df['campaign_summary'] = "No summary available"
        
        # Drop columns that are irrelevant or highly unique (like URLs, creator details, etc.)
        drop_columns = ['creator', 'profile', 'photo', 'slug', 'urls', 'video', 'category', 'blurb']
        df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)
        
        # Remove duplicate records if any
        df.drop_duplicates(inplace=True)
        
        # Handle missing values (Fill numerical with median, categorical with mode)
        for col in df.select_dtypes(include=['number']).columns:
            df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)
        
        # Select only relevant features for prediction
        selected_features = [
            'launched_at', 'id', 'category_id', 'category_name', 'campaign_summary', 'backers_count', 'usd_pledged',
            'converted_pledged_amount', 'staff_pick', 'fx_rate', 'percent_funded', 'goal', 'country', 'currency', 'state'
        ]
        df = df[[col for col in selected_features if col in df.columns]]
        
        return df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Get list of all CSV files in dataset directory
files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".csv")]

# Process all datasets using multithreading
cleaned_dfs = []
with ThreadPoolExecutor(max_workers=5) as executor:
    cleaned_dfs = list(executor.map(clean_kickstarter_data, files))

# Filter out None values
cleaned_dfs = [df for df in cleaned_dfs if df is not None and not df.empty]

# Ensure there is data before merging
if cleaned_dfs:
    final_cleaned_data = pd.concat(cleaned_dfs, ignore_index=True)
    
    # Save the cleaned dataset
    cleaned_file_path = os.path.join(dataset_dir, "Kickstarter_Cleaned.csv")
    final_cleaned_data.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved to {cleaned_file_path}")
else:
    print("No valid data to save.")
