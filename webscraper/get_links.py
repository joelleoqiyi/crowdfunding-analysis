from selenium.webdriver.common.by import By

import time
import random

class LinkScraper:

    def __init__(self, driver, url="https://www.kickstarter.com/discover/advanced?state=live&category_id=16&sort=magic&seed=2898810&page=1"):
        # self.path_to_chrome = r"C:\Users\65845\AppData\Local\Google\Chrome\User Data"

        self.__driver = driver
        self.__url = url
    
    def iterate_pages(self):
        for i in range(1, 28):
            self.__driver.get(self.__url + str(i))
            page_links = self.__scrape_page()
            self.save_links_to_file(page_links, f"links{i}.txt")
            time.sleep(random.uniform(10, 20))

    def __scrape_page(self):
        print(self.__driver.page_source[:1000])
        print("Waiting for elements to load...")
        time.sleep(random.uniform(5, 10))

        elements = self.__driver.find_elements(By.XPATH, "//a[contains(@class, 'project-card__title')]")
        print(f"Total elements found: {len(elements)}")
        # Extract the links and text content
        links = [(element.text, element.get_attribute("href")) for element in elements]
        return links
    
    def save_links_to_file(self, links, filename):
        with open(filename, 'a', encoding='utf-8') as file:
            for _, url in links:
                file.write(f"{url}\n")
