from scraper import Scraper
from get_links import LinkScraper
from comment_updates_scraper import CommentsUpdatesScraper
import pandas as pd
import time

scraper = Scraper()
link_scraper = LinkScraper(scraper.get_driver())
link_scraper.iterate_pages()

with open("links.txt", "r") as file:
    urls = file.readlines()

comments_updates_scraper =  CommentsUpdatesScraper(scraper.get_driver())

data = {}

for link in urls:
    scraper.change_url(link)
    updates_count = CommentsUpdatesScraper.get_updates()
    comments_count = CommentsUpdatesScraper.get_comments()
    updates_content = CommentsUpdatesScraper.extract_updates_content()
    comments_content = CommentsUpdatesScraper.extract_comments_content()
    data[link] = {
        "comments": comments_content,
        "updates": updates_content
    }
    time.sleep(30)

df = pd.DataFrame(data)
df.to_csv("kickstarter_details.csv", index=False)


