import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import random
import time
import logging
from fake_useragent import UserAgent
import sys
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def random_sleep(min_time=1, max_time=3):
    time.sleep(random.uniform(min_time, max_time))

def human_like_scroll(browser, element=None):
    """Scroll in a more human-like way with random pauses"""
    total_height = browser.execute_script("return document.body.scrollHeight")
    viewport_height = browser.execute_script("return window.innerHeight")
    current_position = 0
    
    while current_position < total_height:
        scroll_step = random.randint(100, 400)  # Random scroll amount
        current_position += scroll_step
        browser.execute_script(f"window.scrollTo(0, {current_position});")
        random_sleep(0.5, 1.5)  # Random pause between scrolls

def move_mouse_randomly(browser, element=None):
    """Move mouse in a human-like pattern"""
    try:
        action = ActionChains(browser)
        if element:
            action.move_to_element(element)
        else:
            action.move_by_offset(0, 50)  # Just move down a bit
        action.perform()
    except:
        pass  # Ignore mouse movement errors
    random_sleep(0.1, 0.3)

def get_browser():
    """Initialize undetected-chromedriver with anti-detection measures"""
    options = uc.ChromeOptions()
    ua = UserAgent()
    
    options.add_argument('--headless')  # Add headless mode
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument(f'user-agent={ua.random}')
    options.add_argument('--window-size=1920,1080')
    
    browser = uc.Chrome(options=options)
    
    # Additional stealth measures
    browser.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": ua.random
    })
    
    # Modify navigator.webdriver flag
    browser.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return browser

def wait_and_find_element(browser, by, selector, timeout=10):
    """Wait for and find element with better error handling"""
    try:
        # Wait for element to be both present and clickable
        element = WebDriverWait(browser, timeout).until(
            EC.element_to_be_clickable((by, selector))
        )
        return element
    except Exception as e:
        logger.error(f"Error finding element {selector}: {str(e)}")
        return None

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
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def extract_updates_content(browser):
    updates = []
    scraped_updates = set()  # Set to keep track of scraped updates

    try:
        # Wait for page to be fully loaded
        logger.info("Waiting for page to load completely...")
        WebDriverWait(browser, 10).until(
            lambda x: x.execute_script("return document.readyState") == "complete"
        )
        logger.info("Page loaded completely")
        
        # Get base URL for the project
        base_url = browser.current_url.rstrip('/')
        updates_url = base_url + "/posts"
        
        # First try to find the updates count to verify updates exist
        try:
            logger.info("Looking for updates count...")
            updates_count = browser.find_element(By.CSS_SELECTOR, "a[href$='/posts'] span").text
            updates_count = int(''.join(filter(str.isdigit, updates_count)))
            logger.info(f"Found updates count: {updates_count}")
        except Exception as e:
            logger.error(f"Error finding updates count: {str(e)}")
            return updates

        # Navigate to updates page
        logger.info("Navigating to updates page...")
        browser.get(updates_url)
        random_sleep(3, 5)

        # Process updates until we've found them all or hit a limit
        max_attempts = updates_count + 2  # Add buffer for safety
        attempt = 0
        
        while len(updates) < updates_count and attempt < max_attempts:
            attempt += 1
            logger.info(f"\nAttempt {attempt} to find updates. Found so far: {len(updates)}")
            
            try:
                # Wait for the updates container to be present
                WebDriverWait(browser, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.bg-grey-100.pt6"))
                )
                
                # Find the main updates container
                updates_container = browser.find_element(By.CSS_SELECTOR, "div.bg-grey-100.pt6")
                
                # Find all grid-container divs within the updates container
                update_containers = updates_container.find_elements(By.CSS_SELECTOR, "div.grid-container")
                
                if not update_containers:
                    logger.info("No update containers found")
                    break
                
                # Process each container
                for container in update_containers:
                    try:
                        # Extract update number and check if already scraped
                        try:
                            update_number = container.find_element(By.CSS_SELECTOR, "span.type-13.soft-black_50.text-uppercase").text.strip()
                        except:
                            continue  # Skip if we can't find the update number
                        
                        if update_number in scraped_updates:
                            logger.info(f"Skipping already scraped update: {update_number}")
                            continue
                        
                        # Extract other metadata before clicking
                        try:
                            title = container.find_element(By.CSS_SELECTOR, "h2.kds-heading.mb3").text.strip()
                        except:
                            title = "Unknown Title"
                        
                        try:
                            creator_div = container.find_element(By.CSS_SELECTOR, "div.pl2")
                            creator = creator_div.text.replace("Creator", "").strip()
                        except:
                            creator = "Unknown"
                            
                        try:
                            date = container.find_element(By.CSS_SELECTOR, "span.type-13.soft-black_50.block-md").text.strip()
                        except:
                            date = None
                        
                        # Find the Read More button within this container
                        try:
                            read_more = container.find_element(By.CSS_SELECTOR, "button.ksr-button.bttn")
                            
                            # Click the read more button
                            logger.info(f"Clicking Read more button for {update_number}: {title}")
                            browser.execute_script("arguments[0].click();", read_more)
                            random_sleep(2, 3)
                            
                            # Wait for the detailed page to load
                            WebDriverWait(browser, 10).until(
                                lambda x: x.execute_script("return document.readyState") == "complete"
                            )
                            
                            # Get content
                            try:
                                content_div = browser.find_element(By.CSS_SELECTOR, "div.rte__content")
                                content = content_div.text.strip()
                                logger.info(f"Found content length: {len(content)} characters")
                            except Exception as e:
                                logger.error(f"Error finding content: {str(e)}")
                                content = ""
                            
                            update_data = {
                                'update_number': update_number,
                                'title': title,
                                'creator': creator,
                                'date': date,
                                'content': content,
                                'url': browser.current_url
                            }
                            updates.append(update_data)
                            scraped_updates.add(update_number)
                            logger.info(f"Successfully extracted update: {title[:50]}...")
                            
                            # Navigate back to the updates list
                            browser.back()
                            random_sleep(2, 3)
                            
                            # Wait for the list page to reload
                            WebDriverWait(browser, 10).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "div.bg-grey-100.pt6"))
                            )
                        except Exception as e:
                            logger.error(f"Error with read more button or content extraction: {str(e)}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error processing update container: {str(e)}")
                        continue
                
                # If we haven't found all updates, try scrolling down to load more
                if len(updates) < updates_count:
                    logger.info("Scrolling to load more updates...")
                    human_like_scroll(browser)
                    random_sleep(2, 3)
                
            except Exception as e:
                logger.error(f"Error in update processing loop: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in extract_updates_content: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

    logger.info(f"Finished processing updates. Found {len(updates)} updates.")
    return updates

def extract_comments_content(browser):
    comments = []
    try:
        # Wait for page to be fully loaded
        WebDriverWait(browser, 10).until(
            lambda x: x.execute_script("return document.readyState") == "complete"
        )
        
        # First try to find the comments count to verify comments exist
        try:
            comments_count = browser.find_element(By.CSS_SELECTOR, "a[href$='/comments'] span").text
            logger.info(f"Found comments count: {comments_count}")
        except:
            logger.info("No comments found")
            return comments

        # Try to navigate to comments page directly
        current_url = browser.current_url
        comments_url = current_url + "/comments"
        browser.get(comments_url)
        random_sleep(3, 5)

        # Scroll to load all comments
        human_like_scroll(browser)
        random_sleep(2, 3)
        
        # Find all comments
        comment_elements = browser.find_elements(By.CSS_SELECTOR, ".comment")
        logger.info(f"Found {len(comment_elements)} comment elements")

        for comment in comment_elements:
            try:
                author = comment.find_element(By.CSS_SELECTOR, ".author").text.strip()
                date = comment.find_element(By.CSS_SELECTOR, "time").get_attribute('datetime')
                content = comment.find_element(By.CSS_SELECTOR, ".comment-body").text.strip()

                if author and date and content:
                    comments.append({
                        'author': author,
                        'date': date,
                        'content': content
                    })
                    logger.info(f"Extracted comment from: {author}")
            except Exception as e:
                logger.error(f"Error extracting comment: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in extract_comments_content: {str(e)}")

    return comments

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

if __name__ == "__main__":
    # Test with a single project
    test_url = "https://www.kickstarter.com/projects/genmitsu/cubiko"
    result = scrape_project(test_url)
    
    logger.info("\nFinal results:")
    logger.info(f"URL: {result['url']}")
    logger.info(f"Updates count: {result['updates']['count']}")
    
    if result['updates']['content']:
        logger.info("\nFirst update example:")
        logger.info(result['updates']['content'][0])
    
    # Uncomment to scrape multiple technology projects
    # projects = scrape_technology_projects()
    # logger.info(f"Scraped {len(projects)} technology projects") 