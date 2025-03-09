import pandas as pd
import os

def save_to_csv(df, filename):
    """Appends new data to the existing CSV and sorts by campaigns (based on an id assigned to them)."""
    
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(
            subset = ["campaign_id", "scrape_date"], keep = "last"
        )

    # Remove duplicates and sort by launch date
    df = df.sort_values(by=["campaign_id", "scrape_date"], ascending=False)

    df.to_csv(filename, index=False)
    print(f"Data successfully saved to {filename}. Total projects: {len(df)}")
