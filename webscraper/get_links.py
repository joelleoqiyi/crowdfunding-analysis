from selenium.webdriver.common.by import By

import time
import random

class LinkScraper:

    def __init__(self, driver, url="https://www.kickstarter.com/discover/advanced?state=live&category_id=16&sort=magic&seed=2898810&page="):
        # self.path_to_chrome = r"C:\Users\65845\AppData\Local\Google\Chrome\User Data"

        # options = Options()
        # options.add_argument("--headless")  # Run without opening a browser window
        # options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent Selenium detection
        # options.add_argument("user-data-dir=" + self.path_to_chrome)
        # options.add_argument("profile-directory=Default")  # Use your Chrome's main profile
        # options.add_argument("--no-sandbox")
        # options.add_argument("--disable-dev-shm-usage")
        # options.add_argument("--window-size=1920,1080")
        # options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        # self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        # wait = WebDriverWait(self.driver, 20)
        # self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        self.__driver = driver
        self.__url = url
    
    def iterate_pages(self):
        for i in range(1, 28):
            self.__driver.get(self.__url + str(i))
            page_links = self.__scrape_page()
            self.save_links_to_file(page_links, 'links.txt')
            time.sleep(random.randint(5, 10))

    def __scrape_page(self):
        print(self.__driver.page_source[:1000])
        print("Waiting for elements to load...")
        time.sleep(5)

        elements = self.__driver.find_elements(By.XPATH, "//a[contains(@class, 'project-card__title')]")
        print(f"Total elements found: {len(elements)}")
        # Extract the links and text content
        links = [(element.text, element.get_attribute("href")) for element in elements]
        return links
    
    def save_links_to_file(self, links, filename):
        with open(filename, 'a', encoding='utf-8') as file:
            for _, url in links:
                file.write(f"{url}\n")
