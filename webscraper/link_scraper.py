from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import os

class LinkScraper:
    def __init__(self, driver, url="https://www.kickstarter.com/discover/advanced?state=live&category_id=16&sort=magic&seed=2898810&page="):
        self.__driver = driver
        self.__url = url
    
    def iterate_pages(self, num_pages=2):
        all_links = []
        for i in range(1, num_pages + 1):
            print(f"Scraping page {i}...")
            self.__driver.get(self.__url + str(i))
            page_links = self.__scrape_page()
            self.save_links_to_file(page_links, 'scraped_links.txt')
            all_links.extend(page_links)
            # Random delay between pages
            delay = random.uniform(3, 5)
            print(f"Waiting {delay:.1f} seconds before next page...")
            time.sleep(delay)
        return all_links

    def __scrape_page(self):
        print("Waiting for elements to load...")
        try:
            # Wait for the project cards to be visible
            wait = WebDriverWait(self.__driver, 10)
            wait.until(EC.presence_of_all_elements_located((By.XPATH, "//a[contains(@class, 'project-card__title')]")))
            
            # Scroll down to load all content
            self.__driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for any dynamic content to load
            
            elements = self.__driver.find_elements(By.XPATH, "//a[contains(@class, 'project-card__title')]")
            print(f"Found {len(elements)} projects on this page")
            
            # Extract the links and text content
            links = [(element.text, element.get_attribute("href")) for element in elements]
            return links
            
        except Exception as e:
            print(f"Error scraping page: {e}")
            return []
    
    def save_links_to_file(self, links, filename):
        try:
            with open(filename, 'a', encoding='utf-8') as file:
                for _, url in links:
                    if url:  # Only write non-empty URLs
                        # Clean the URL by removing query parameters
                        clean_url = url.split('?')[0]
                        file.write(f"{clean_url}\n")
            print(f"Saved {len(links)} links to {filename}")
        except Exception as e:
            print(f"Error saving to file: {e}")

def setup_driver():
    """Setup Chrome WebDriver with appropriate options"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in headless mode
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def main():
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Clear existing scraped_links.txt if it exists
    if os.path.exists('scraped_links.txt'):
        os.remove('scraped_links.txt')
    
    driver = setup_driver()
    try:
        print("Starting to scrape Kickstarter projects...")
        scraper = LinkScraper(driver)
        scraper.iterate_pages(num_pages=2)  # Scrape first 2 pages
        print("Scraping completed!")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main() 