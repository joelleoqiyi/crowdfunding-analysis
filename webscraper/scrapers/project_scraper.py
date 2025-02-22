import json
import os
from datetime import datetime
import logging
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils.browser_utils import random_sleep, get_browser
from scrapers.update_scraper import extract_updates_content

logger = logging.getLogger(__name__)

def save_to_json(data, project_url):
    """Save scraped data to a JSON file"""
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create a scraped_data directory in the same directory as the script
        data_dir = os.path.join(script_dir, 'scraped_data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Create a filename based on project URL and timestamp
        project_name = project_url.split('/')[-1]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(data_dir, f'{project_name}_{timestamp}.json')
        
        # Save the data
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data saved to {filename}")
        
        return filename
    except Exception as e:
        logger.error(f"Error saving data to JSON: {str(e)}")
        return None

def scrape_project(url):
    """Scrape a single project"""
    browser = None
    try:
        browser = get_browser()
        logger.info(f"Processing project: {url}")
        
        browser.get(url)
        # Wait for page to be fully loaded
        WebDriverWait(browser, 20).until(
            lambda x: x.execute_script("return document.readyState") == "complete"
        )
        random_sleep(3, 5)  # Wait for page load and potential Cloudflare check
        
        # Initial scroll to simulate human behavior
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight/4);")
        random_sleep(2, 3)
        
        updates_content = extract_updates_content(browser)
        logger.info(f"Extracted {len(updates_content)} updates")
        
        result = {
            'url': url,
            'updates': {
                'count': len(updates_content),
                'content': updates_content
            }
        }
        
        # Save final result only once
        logger.info("Saving final result...")
        saved_file = save_to_json(result, url)
        logger.info(f"Saved to: {saved_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}")
        return {
            'url': url,
            'updates': {'count': 0, 'content': []},
            'error': str(e)
        }
    finally:
        if browser:
            browser.quit()

def scrape_technology_projects(start_page=1, max_pages=5):
    """Scrape multiple technology projects"""
    base_url = "https://www.kickstarter.com/discover/categories/technology"
    projects = []
    
    browser = get_browser()
    try:
        for page in range(start_page, start_page + max_pages):
            url = f"{base_url}?page={page}"
            browser.get(url)
            random_sleep(3, 5)
            
            # Find all project links
            project_links = browser.find_elements(By.CSS_SELECTOR, "a.project-title")
            project_urls = [link.get_attribute('href') for link in project_links]
            
            for project_url in project_urls:
                project_data = scrape_project(project_url)
                projects.append(project_data)
                random_sleep(5, 10)  # Pause between projects
                
    except Exception as e:
        logger.error(f"Error in scrape_technology_projects: {str(e)}")
    finally:
        browser.quit()
    
    return projects 
