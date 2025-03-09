import pandas as pd
from datetime import datetime, timezone
import hashlib

def format_unix_timestamp(unix_timestamp):
    """Converts Unix timestamp to a human-readable date format."""
    if isinstance(unix_timestamp, int):
        return datetime.fromtimestamp(unix_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return "N/A"

def calculate_days_left(deadline):
    """Calculates the number of days left until the campaign deadline."""
    if deadline == "N/A":
        return "N/A"

    try:
        deadline_date = datetime.strptime(deadline, "%Y-%m-%d")
        today_date = datetime.now(datetime.timezone.utc)
        return (deadline_date - today_date).days
    except ValueError:
        return "N/A"
    
def generate_campaign_id(title):
    """Generate a unique ID for each campaign based on its title."""
    return hashlib.md5(title.encode()).hexdigest()[:10]  # Shorter hash for readability


def process_scraped_data(raw_data):
    """Converts scraped data into a DataFrame and formats it."""
    df = pd.DataFrame(raw_data)

    # Generate a unique campaign ID based on title
    df["campaign_id"] = df["name"].apply(generate_campaign_id)
    # Add today's date to track daily updates
    df["scrape_date"] = datetime.today().strftime('%Y-%m-%d')

    # Convert timestamps to readable dates
    df["created_at"] = df["created_at"].apply(format_unix_timestamp)
    df["launched_at"] = df["launched_at"].apply(format_unix_timestamp)
    df["state_changed_at"] = df["state_changed_at"].apply(format_unix_timestamp)
    df["deadline"] = df["deadline"].apply(format_unix_timestamp)
    df["days_left"] = df["deadline"].apply(calculate_days_left)
    
    return df
