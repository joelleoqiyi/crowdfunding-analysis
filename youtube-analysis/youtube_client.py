# youtube_client.py
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_config import YOUTUBE_API_KEY

class YouTubeClient:
    def __init__(self):
        if not YOUTUBE_API_KEY:
            raise RuntimeError("YOUTUBE_API_KEY must be set")
        self.youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    def get_video_comments(self, video_id: str):
        comments = []
        request = self.youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        )
        print("Youtube comment request", request)
        while request:
            try:
                resp = request.execute()
                print("Youtube comment request response", resp)
            except HttpError as e:
                msg = str(e)
                # skip videos with comments disabled
                if e.resp.status == 403 and "commentsDisabled" in msg:
                    break
                # propagate quota errors
                if "quotaExceeded" in msg or "rateLimitExceeded" in msg:
                    raise
                raise
            for item in resp.get("items", []):
                snip = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "youtube_vid": video_id,
                    "comment": snip.get("textOriginal", ""),
                    "author":  snip.get("authorDisplayName", "")
                })
            request = self.youtube.commentThreads().list_next(request, resp)
        return comments

    def scrape_comments(self, query: str, max_results: int = 10):
        print("Searching youtube video now: ", query)
        # search for videos matching the campaign name
        search_resp = self.youtube.search().list(
            q=query, part="id", type="video", maxResults=max_results
        ).execute()
        print("Search response: ", search_resp)
        vids = [item["id"]["videoId"] for item in search_resp.get("items", [])]
        all_comments = []
        for vid in vids:
            all_comments.extend(self.get_video_comments(vid))
        print("Video comments: ", all_comments)
        return all_comments