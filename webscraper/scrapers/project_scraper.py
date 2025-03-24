import json
import os
from datetime import datetime
import logging
import traceback
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils.browser_utils import random_sleep, get_browser
from scrapers.update_scraper import extract_updates_content, extract_campaign_details
import random

logger = logging.getLogger(__name__)

def save_to_json(data, project_url):
    """Save scraped data to a JSON file"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Use only one directory for all projects
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
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def scrape_project(url):
    """Scrape a single project including campaign details and updates"""
    browser = None
    try:
        browser = get_browser()
        logger.info(f"Processing project: {url}")
        
        # Navigate to the project's main page first
        browser.get(url)
        WebDriverWait(browser, 20).until(
            lambda x: x.execute_script("return document.readyState") == "complete"
        )
        random_sleep(3, 5)  
        
        # Scroll down a bit to ensure content loads
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight/4);")
        random_sleep(2, 3)
        
        # Step 1: Extract campaign details from the main page
        logger.info("Extracting campaign details from main page...")
        campaign_details = extract_campaign_details(browser)
        
        # Step 2: Navigate to updates page and extract updates
        logger.info("Navigating to updates page...")
        
        # Properly construct the updates URL
        clean_base_url = url.rstrip('/')  # Remove any trailing slash
        updates_url = f"{clean_base_url}/posts"
        
        logger.info(f"Navigating directly to updates URL: {updates_url}")
        browser.get(updates_url)
        random_sleep(3, 5)
        
        # Wait for updates page to load
        WebDriverWait(browser, 15).until(
            lambda x: x.execute_script("return document.readyState") == "complete"
        )
        
        # Now extract updates from the updates page
        logger.info("Extracting updates content...")
        updates_data = extract_updates_content(browser)
        updates_content = updates_data.get('updates', [])
        
        logger.info(f"Extracted {len(updates_content)} updates")
        
        # Combine all data into a single result
        result = {
            'url': url,
            'scraped_at': datetime.now().isoformat(),
            'campaign_details': campaign_details,
            'updates': {
                'count': len(updates_content),
                'content': updates_content
            }
        }
        
        logger.info("Saving final result...")
        saved_file = save_to_json(result, url)
        logger.info(f"Saved to: {saved_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {url}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            'url': url,
            'scraped_at': datetime.now().isoformat(),
            'campaign_details': {},
            'updates': {'count': 0, 'content': []},
            'error': str(e)
        }
    finally:
        if browser:
            try:
                browser.quit()
            except Exception as quit_e:
                logger.error(f"Error closing browser: {str(quit_e)}")

def scrape_technology_projects(start_page=1, max_pages=5):
    """Scrape multiple technology projects"""
    # Base URL for technology projects - can include successful or failed
    base_url = "https://www.kickstarter.com/discover/categories/technology"
    projects = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create directory for data storage - single directory for all projects
    data_dir = os.path.join(script_dir, 'scraped_data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Setup results file
    results_file = os.path.join(
        data_dir, 
        f'technology_projects_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    
    browser = get_browser()
    try:
        for page in range(start_page, start_page + max_pages):
            url = f"{base_url}?page={page}"
            logger.info(f"Processing page {page} of {start_page + max_pages - 1}")
            browser.get(url)
            random_sleep(3, 5)
            
            # Wait for page to load completely
            WebDriverWait(browser, 15).until(
                lambda x: x.execute_script("return document.readyState") == "complete"
            )
            
            # Find project links - try multiple selectors to improve reliability
            project_links = []
            selectors_to_try = [
                "a.project-title",  # Standard selector
                "div.js-track-project-card a",  # New layout
                "a[data-test-id='project-card']",  # Another possible format
                "div.grid-container a",  # Alternative format
                "a[href*='/projects/']"  # Any link containing project path
            ]
            
            for selector in selectors_to_try:
                try:
                    logger.info(f"Trying to find project links with selector: {selector}")
                    links = browser.find_elements(By.CSS_SELECTOR, selector)
                    if links:
                        project_links = links
                        logger.info(f"Found {len(links)} projects using selector: {selector}")
                        break
                except Exception as selector_e:
                    logger.error(f"Error with selector {selector}: {str(selector_e)}")
            
            # Extract URLs from links - make sure we're only getting project links
            project_urls = []
            for link in project_links:
                try:
                    href = link.get_attribute('href')
                    if href and '/projects/' in href:
                        # Skip links to updates, rewards, etc.
                        if any(x in href for x in ['/posts', '/rewards', '/comments', '/faqs']):
                            continue
                        # Remove any tracking parameters
                        clean_url = href.split('?')[0]
                        if clean_url not in project_urls:  # Avoid duplicates
                            project_urls.append(clean_url)
                except Exception as e:
                    logger.error(f"Error extracting link: {str(e)}")
            
            # Log results
            logger.info(f"Found {len(project_urls)} unique project URLs on page {page}")
            
            if not project_urls:
                logger.warning(f"No project links found on page {page}. Continuing to next page.")
                continue
            
            # Process each project
            for i, project_url in enumerate(project_urls, 1):
                logger.info(f"\n{'='*30}")
                logger.info(f"Processing project {i}/{len(project_urls)} on page {page}")
                logger.info(f"URL: {project_url}")
                logger.info(f"{'='*30}")
                
                try:
                    # Scrape project
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
                    logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Sleep between projects to avoid rate limiting
                sleep_time = random.randint(5, 10)
                logger.info(f"Sleeping for {sleep_time} seconds before next project...")
                random_sleep(5, 10)
            
            # Sleep between pages to avoid rate limiting
            if page < start_page + max_pages - 1:  # Don't sleep after the last page
                logger.info(f"Completed page {page}. Sleeping before next page...")
                random_sleep(15, 30)  # Longer sleep between pages
                
    except Exception as e:
        logger.error(f"Error in scrape_technology_projects: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        try:
            browser.quit()
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")
    
    # Final log message
    logger.info(f"\n{'='*50}")
    logger.info(f"Finished scraping {len(projects)} projects")
    logger.info(f"Results saved to {results_file}")
    logger.info(f"{'='*50}")
    
    return projects
