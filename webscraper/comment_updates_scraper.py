from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from utils import get_comments_link
import time
import random

class CommentsUpdatesScraper:
    def __init__(self, driver):
        self.__driver = driver

    def get_comments_texts(self, url: str):
        self.__driver.get(url)

        time.sleep(random.uniform(5, 10))

        more_comments_buttons = self.__driver.find_elements(By.CLASS_NAME, 'ksr-button')

        for button in more_comments_buttons:
            try:
                self.__driver.execute_script("arguments[0].scrollIntoView();", button)  # Scroll to the button
                time.sleep(random.uniform(1, 2))
                self.__driver.execute_script("arguments[0].click();", button)
                time.sleep(random.uniform(5, 10))  # Allow time for content to load after clicking
            except Exception as e:
                print(f"Could not click button: {e}")


        for _ in range(5):  # Adjust this based on how many comments you need
            self.__driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
            time.sleep(random.uniform(3, 5))  # Allow time for comments to load

        # Extract all comment elements
        comment_elements = self.__driver.find_elements(By.CLASS_NAME, "data-comment-text")

        # Extract and return text from each comment
        return [comment.text.strip() for comment in comment_elements if comment.text.strip()]
    
    def get_comments_into_txt(self, url: str):
        url = get_comments_link(url)
        comments = self.get_comments_texts(url)
        try:
            project_name = url.split("/projects/")[1].split("/")[0]
        except IndexError:
            project_name = "kickstarter_project"
        filename = f"{project_name}-comments.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n\n")
            for comment in comments:
                f.write(comment + "\n\n")
        
        print(f"Comments saved to {filename}")