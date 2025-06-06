from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random
import os
from utils.browser_utils import random_sleep, human_like_scroll, move_mouse_randomly, get_browser

class LinkScraper:
    def __init__(self, driver, url=None):
        self.__driver = driver
        
        # URL for live projects - technology category (16)
        # Changed sort to end_date for better analysis - provides projects at different stages of completion
        self.__url = "https://www.kickstarter.com/discover/advanced?state=live&category_id=16&woe_id=0&sort=end_date&seed=2902974&page="
        
        # If a specific URL is provided, use that instead
        if url:
            self.__url = url
    
    def iterate_pages(self, num_pages=2):
        all_links = []
        for i in range(1, num_pages + 1):
            print(f"Scraping page {i}...")
            self.__driver.get(self.__url + str(i))
            page_links = self.__scrape_page()
            self.save_links_to_file(page_links, 'unscraped_links.txt')
            all_links.extend(page_links)
            # Random delay between pages
            delay = random.uniform(3, 5)
            print(f"Waiting {delay:.1f} seconds before next page...")
            time.sleep(delay)
        return all_links

    def __scrape_page(self):
        print("Waiting for elements to load...")
        try:
            # Initial random wait
            random_sleep(2, 4)
            
            # Scroll in a human-like way
            human_like_scroll(self.__driver)
            
            # Move mouse randomly
            move_mouse_randomly(self.__driver)
            
            wait = WebDriverWait(self.__driver, 15)
            project_cards = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.js-react-proj-card")))
            
            print(f"Found {len(project_cards)} projects on this page")
            
            # Extract links with random mouse movements
            links = []
            for card in project_cards:
                try:
                    move_mouse_randomly(self.__driver, card)
                    
                    title_element = card.find_element(By.CSS_SELECTOR, "a.project-card__title")
                    title = title_element.text
                    href = title_element.get_attribute("href")
                    
                    # Get funding amount to verify sorting
                    amount = "N/A"
                    try:
                        money_element = card.find_element(By.CSS_SELECTOR, "span.money")
                        amount = money_element.text
                    except:
                        pass
                    print(f"Project: {title} - Amount: {amount}")
                    
                    if href and title:
                        updates_count = 0
                        try:
                            updates_element = card.find_element(By.CSS_SELECTOR, "a[href$='/posts'] span")
                            updates_count = int(''.join(filter(str.isdigit, updates_element.text)))
                        except:
                            pass
                        links.append((title, href, updates_count))
                        
                    random_sleep(0.1, 0.3)
                except Exception as e:
                    print(f"Error processing card: {str(e)}")
                    continue
                    
            return links
            
        except Exception as e:
            print(f"Error scraping page: {str(e)}")
            return []
    
    def save_links_to_file(self, links, filename):
        try:
            # Sort links by update count in descending order
            sorted_links = sorted(links, key=lambda x: x[2], reverse=True)
            
            with open(filename, 'a', encoding='utf-8') as file:
                for _, url, count in sorted_links:
                    if url:  
                        # Clean the URL by removing query parameters
                        clean_url = url.split('?')[0]
                        file.write(f"{clean_url}\t{count}\n")
            print(f"Saved {len(links)} links to {filename}")
        except Exception as e:
            print(f"Error saving to file: {e}")

def setup_driver():
    """Setup undetected-chromedriver with anti-detection measures"""
    return get_browser()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Read existing links to avoid duplicates
    existing_links = set()
    
    # Read from the previously scraped links file
    if os.path.exists('previously_scraped_links.txt'):
        with open('previously_scraped_links.txt', 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    existing_links.add(parts[0])
        print(f"Loaded {len(existing_links)} links from previously_scraped_links.txt to avoid duplicates")
    
    # If unscraped_links.txt exists, read it but DON'T clear it
    # This way we append new links to any existing unscraped links
    if os.path.exists('unscraped_links.txt'):
        with open('unscraped_links.txt', 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    existing_links.add(parts[0])
        print(f"Also considering {len(existing_links)} links from unscraped_links.txt")
    
    driver = setup_driver()
    try:
        print("Starting to scrape live Kickstarter technology projects (sorted by end date)...")
        
        scraper = LinkScraper(driver)
        live_links = []
        page = 13
        target_live_links = 50  # Target 50 new unique live links
        consecutive_failures = 0
        max_pages = 30
        
        print(f"Avoiding {len(existing_links)} existing links")
        print(f"Targeting {target_live_links} new unique live technology links")
        
        while len(live_links) < target_live_links and page <= max_pages and consecutive_failures < 3:
            print(f"Scraping live projects page {page}...")
            driver.get(scraper._LinkScraper__url + str(page))
            page_links = scraper._LinkScraper__scrape_page()
            
            # Filter out already scraped links
            new_links = [(title, href, updates) for title, href, updates in page_links 
                         if href.split('?')[0] not in existing_links]
            
            if new_links:
                consecutive_failures = 0
                live_links.extend(new_links)
                # Add to existing links to avoid duplicates in future pages
                existing_links.update(href.split('?')[0] for _, href, _ in new_links)
                print(f"Found {len(new_links)} new live links on page {page}")
                print(f"Total new live links so far: {len(live_links)}/{target_live_links}")
            else:
                consecutive_failures += 1
                print(f"No new live links found on page {page}. Consecutive failures: {consecutive_failures}")
            
            # Save live links after each page
            scraper.save_links_to_file(new_links, 'unscraped_links.txt')
            
            page += 1
            # Random delay between pages
            delay = random.uniform(3, 5)
            print(f"Waiting {delay:.1f} seconds before next page...")
            time.sleep(delay)
        
        print(f"Finished scraping live technology projects. Found {len(live_links)} new unique links.")
        print(f"Projects are sorted by end date, providing a mix of campaigns at different stages for better analysis.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main() 