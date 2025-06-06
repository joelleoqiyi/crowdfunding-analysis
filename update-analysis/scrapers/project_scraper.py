import json
import os
from datetime import datetime
import logging
import traceback
import sys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Get absolute path to the update-analysis directory (parent of scrapers)
update_analysis_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add utils directory to path if not already added
utils_dir = os.path.join(update_analysis_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.append(utils_dir)

# Import browser utils directly
from browser_utils import random_sleep, get_browser

# Import from the same directory
from update_scraper import extract_updates_content

logger = logging.getLogger(__name__)

def save_to_json(data, project_url):
    """Save scraped data to a JSON file"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        data_dir = os.path.join(script_dir, 'scraped_data')
        os.makedirs(data_dir, exist_ok=True)
        
        project_name = project_url.split('/')[-1]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(data_dir, f'{project_name}_{timestamp}.json')
        
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
        print("THIS IS A BROWSER: ", browser)
        logger.info(f"Processing project: {url}")
        
        # URL should already be properly formatted with /posts from server.py
        # Navigate directly to the provided URL
        browser.get(url)
        
        # Wait for page to load
        WebDriverWait(browser, 20).until(
            lambda x: x.execute_script("return document.readyState") == "complete"
        )
        random_sleep(3, 5)  
        
        # Initial scroll to load content
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight/4);")
        random_sleep(2, 3)
        
        # Additional scroll to ensure more content is loaded
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        random_sleep(2, 3)
        
        # Final scroll to load all content
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        random_sleep(2, 3)
        
        # Extract all data using the update_scraper
        logger.info("Calling extract_updates_content function...")
        scraped_data = extract_updates_content(browser)
        
        if not scraped_data:
            logger.error("No data returned from extract_updates_content")
            return {
                'url': url,
                'campaign_details': {},
                'updates': {'count': 0, 'content': []},
                'error': 'No data returned from update scraper'
            }
        
        # Extract updates content
        updates_content = scraped_data.get('updates', [])
        campaign_details = scraped_data.get('campaign_details', {})
        
        # Log results
        logger.info(f"Extracted campaign details: {', '.join([f'{k}: {v}' for k, v in campaign_details.items()])}")
        logger.info(f"Extracted {len(updates_content)} updates")
        
        # Create result structure
        result = {
            'url': url,
            'campaign_details': campaign_details,
            'updates': {
                'count': len(updates_content),
                'content': updates_content
            }
        }
        
        # Save to JSON for debugging/reference
        logger.info("Saving scraped data to JSON...")
        saved_file = save_to_json(result, url)
        logger.info(f"Saved to: {saved_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'url': url,
            'campaign_details': {},
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
    results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               'scraped_data', 
                               f'technology_projects_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    browser = get_browser()
    print("THIS IS A BROWSER: ", browser)

    try:
        for page in range(start_page, start_page + max_pages):
            url = f"{base_url}?page={page}"
            logger.info(f"Processing page {page} of {start_page + max_pages - 1}")
            browser.get(url)
            random_sleep(3, 5)
            
            project_links = browser.find_elements(By.CSS_SELECTOR, "a.project-title")
            project_urls = [link.get_attribute('href') for link in project_links]
            
            logger.info(f"Found {len(project_urls)} projects on page {page}")
            
            for i, project_url in enumerate(project_urls, 1):
                logger.info(f"Processing project {i}/{len(project_urls)} on page {page}")
                logger.info(f"URL: {project_url}")
                
                try:
                    project_data = scrape_project(project_url)
                    projects.append(project_data)
                    
                    # Save intermediate results after each project
                    with open(results_file, 'w', encoding='utf-8') as f:
                        json.dump(projects, f, indent=2, ensure_ascii=False)
                    
                    # Log some info about the project
                    if 'campaign_details' in project_data and project_data['campaign_details']:
                        campaign_details = project_data['campaign_details']
                        logger.info(f"Campaign details: Pledged: {campaign_details.get('pledged_amount', 'N/A')}, " +
                                   f"Goal: {campaign_details.get('funding_goal', 'N/A')}, " +
                                   f"Backers: {campaign_details.get('backers_count', 'N/A')}")
                    
                    updates_count = project_data['updates']['count'] if 'updates' in project_data else 0
                    logger.info(f"Updates count: {updates_count}")
                    
                except Exception as e:
                    logger.error(f"Error processing project {project_url}: {str(e)}")
                
                random_sleep(5, 10)
                
    except Exception as e:
        logger.error(f"Error in scrape_technology_projects: {str(e)}")
    finally:
        browser.quit()
    
    logger.info(f"Finished scraping {len(projects)} technology projects")
    logger.info(f"Results saved to {results_file}")
    
    return projects 