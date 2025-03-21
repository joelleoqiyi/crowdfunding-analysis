from scraper import Scraper
from get_links import LinkScraper
from comment_updates_scraper import CommentsUpdatesScraper
import time

scraper = Scraper()
# link_scraper = LinkScraper(scraper.get_driver())
# link_scraper.iterate_pages()

with open("cleaned_links.txt", "r") as file:
    urls = file.readlines()

comments_updates_scraper =  CommentsUpdatesScraper(scraper)

data = {}

startindex = 205
endindex = 225

for i in range(startindex, endindex):
    print(i)
    link = urls[i].strip()
    comments_updates_scraper.get_comments_into_json(link)
    scraper.random_sleep(3, 6)
