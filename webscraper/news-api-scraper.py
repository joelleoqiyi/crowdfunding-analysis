import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("NEWS_API_KEY")

# Define the query and endpoint
query = "Planck: The World’s Smallest Phone-first SSD with up to 2TB"
url = "https://newsapi.org/v2/everything"

# Define query parameters
params = {
    "q": query,
    "apiKey": api_key
}

# Make the GET request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    articles = data.get("articles", [])
    print(f"Found {len(articles)} articles:")
    for article in articles:
        print("-", article.get("title"))
else:
    print(f"Failed to fetch data: {response.status_code}")
    print(response.text)