from scraper import Scraper
from get_links import LinkScraper
from comment_updates_scraper import CommentsUpdatesScraper
import pandas as pd
import time

scraper = Scraper()
# link_scraper = LinkScraper(scraper.get_driver())
# link_scraper.iterate_pages()

with open("links1.txt", "r") as file:
    urls = file.readlines()

comments_updates_scraper =  CommentsUpdatesScraper(scraper.get_driver())

data = {}

for link in urls:
    print(link)
    comments_updates_scraper.get_comments_into_txt(link)

    time.sleep(30)

