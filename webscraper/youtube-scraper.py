import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("YOUTUBE_API_KEY")

# Define your search query
search_query = "Planck: The World’s Smallest Phone-first SSD with up to 2TB"

# Build the YouTube service
youtube = build("youtube", "v3", developerKey=api_key)

# Search for videos matching your query
search_response = youtube.search().list(
    q=search_query,
    part="id,snippet",
    type="video",
    maxResults=10  # adjust as needed
).execute()

# Extract video IDs from search results
video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
print("Found video IDs:", video_ids)

def get_video_comments(youtube, video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100  # maximum allowed per request
    )
    while request:
        response = request.execute()
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
        request = youtube.commentThreads().list_next(request, response)
    return comments

# Retrieve comments for each video
all_comments = {}
for vid in video_ids:
    print(f"Retrieving comments for video ID: {vid}")
    comments = get_video_comments(youtube, vid)
    all_comments[vid] = comments
    print(f"Found {len(comments)} comments for video ID: {vid}")

# Optionally, print the comments for each video
for vid, comments in all_comments.items():
    print(f"\nComments for video {vid}:")
    for comment in comments:
        print("-", comment)