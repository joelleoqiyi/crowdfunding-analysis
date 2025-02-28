from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver

class Scraper:

    def __init__(self):
        options = webdriver.ChromeOptions()
        #options.add_argument("--headless")  # Run without opening a browser window
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

        self.__driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def get_driver(self):
        return self.__driver
    
    def change_url(self, url):
        self.__driver.get(url)
