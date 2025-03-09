import tweepy
import os

# Replace with your own Bearer Token
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
 
# Initialize the client
client = tweepy.Client(bearer_token=bearer_token)

# The ID of the tweet you want to get comments for
tweet_id = "1890028527539585500"

# Build the query using the conversation_id filter
query = f"conversation_id:{tweet_id}"

# Fetch recent tweets (replies)
response = client.search_recent_tweets(query=query, tweet_fields=["author_id", "created_at", "conversation_id"])

if response.data:
    for tweet in response.data:
        print(f"{tweet.created_at} - {tweet.author_id}: {tweet.text}")
else:
    print("No replies found.")