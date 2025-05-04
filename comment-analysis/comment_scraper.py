from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import os
import json

class CommentsUpdatesScraper:
    def __init__(self, scraper):
        self.__driver = scraper.get_driver()

    def get_comments_texts(self, url: str):
        url = url.rstrip('/')

        if not url.endswith("/comments"):
            url += "/comments"

        
        self.__driver.get(url)

        WebDriverWait(self.__driver, 10).until(
            lambda x: x.execute_script("return document.readyState") == "complete"
        )

        time.sleep(random.uniform(1, 3))

        more_comments_buttons = self.__driver.find_elements(By.CLASS_NAME, "button.ksr-button:not(.theme--support)")

        for button in more_comments_buttons:
            try:
                self.__driver.execute_script("arguments[0].scrollIntoView();", button)  # Scroll to the button
                time.sleep(random.uniform(1, 2))
                self.__driver.execute_script("arguments[0].click();", button)
                time.sleep(random.uniform(3, 5))  # Allow time for content to load after clicking
            except Exception as e:
                print(f"Could not click button: {e}")


        for _ in range(5):  # Adjust this based on how many comments you need
            self.__driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
            time.sleep(random.uniform(3, 5))  # Allow time for comments to load

        # Extract all comment elements
        comment_elements = self.__driver.find_elements(By.CLASS_NAME, "data-comment-text")

        comments_data = []

        for comment in comment_elements:
            try:
                # Locate the parent element of the comment
                parent_element = comment.find_element(By.XPATH, "./ancestor::li[1]")

                # Find the poster name relative to the comment
                poster_element = parent_element.find_element(By.XPATH, ".//div/div[1]/div/div/span[1]/span")

                poster_name = poster_element.text.strip()
                comment_text = comment.text.strip()

                time_element = parent_element.find_element(By.XPATH, ".//div/div[1]/div/div/a/time")
                comment_time = time_element.get_attribute("datetime")

                comment_text = comment.text.strip()

                try:
                    creator_tag = parent_element.find_element(By.XPATH, "./div/div[1]/div/div/span[2]/span")
                    class_attribute = creator_tag.get_attribute("class")

                    # If the span contains the "bg-ksr-green-700" class, it's a creator
                    if "bg-ksr-green-700" in class_attribute:
                        is_creator = True
                    else:
                        is_creator = False
                except:
                    is_creator = False

                if comment_text and poster_name and comment_time:
                    comments_data.append({
                        "poster": poster_name,
                        "comment": comment_text,
                        "timestamp": comment_time,
                        "is_creator": is_creator
                    })
            except Exception as e:
                print(f"Error retrieving comment details: {e}")

        return comments_data
    
    def get_comments_into_txt(self, url: str):
        url = url + "/comments"
        comments = self.get_comments_texts(url)
        
        try:
            project_name = url.split("/projects/")[1].split("/")[0]
        except IndexError:
            project_name = "kickstarter_project"
        
        # Define the directory
        directory = "scraped_data"
        
        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Construct the full file path
        filename = os.path.join(directory, f"{project_name}-comments.txt")
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\n\n")
            for comment in comments:
                f.write(comment + "\n\n")
        
        print(f"Comments saved to {filename}")

    def get_comments_into_json(self, url: str):
        url = url + "/comments"
        comments = self.get_comments_texts(url)

        try:
            project_name = url.split("/projects/")[1].split("/")[1]
        except IndexError:
            project_name = ""

        if project_name == "":
            try:
                project_name = url.split("/projects/")[1].split("/")[0]
            except IndexError:
                project_name = "kickstarter_project"

        # Define the directory
        directory = "scraped_data"

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Construct the full file path
        filename = os.path.join(directory, f"{project_name}-comments.json")

        # Structure the JSON data
        data = {
            "url": url,
            "comments": comments
        }

        # Save data as JSON
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"Comments saved to {filename}")