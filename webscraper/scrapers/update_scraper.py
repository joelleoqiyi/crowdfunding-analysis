from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import traceback
from utils.browser_utils import random_sleep, human_like_scroll

logger = logging.getLogger(__name__)

def extract_updates_content(browser):
    """Extract all updates from a project's updates page"""
    updates = []
    scraped_updates = set()  

    try:
        logger.info("Waiting for page to load completely...")
        WebDriverWait(browser, 10).until(
            lambda x: x.execute_script("return document.readyState") == "complete"
        )
        logger.info("Page loaded completely")
        
        base_url = browser.current_url.rstrip('/')
        updates_url = base_url + "/posts"
        
        try:
            logger.info("Looking for updates count...")
            updates_count = browser.find_element(By.CSS_SELECTOR, "a[href$='/posts'] span").text
            updates_count = int(''.join(filter(str.isdigit, updates_count)))
            logger.info(f"Found updates count: {updates_count}")
        except Exception as e:
            logger.error(f"Error finding updates count: {str(e)}")
            return updates

        logger.info("Navigating to updates page...")
        browser.get(updates_url)
        random_sleep(3, 5)

        max_attempts = updates_count + 2 
        attempt = 0
        
        while len(updates) < updates_count and attempt < max_attempts:
            attempt += 1
            logger.info(f"\nAttempt {attempt} to find updates. Found so far: {len(updates)}")
            
            try:
                WebDriverWait(browser, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.bg-grey-100.pt6"))
                )
                
                updates_container = browser.find_element(By.CSS_SELECTOR, "div.bg-grey-100.pt6")
                
                update_containers = updates_container.find_elements(By.CSS_SELECTOR, "div.grid-container")
                
                if not update_containers:
                    logger.info("No update containers found")
                    break
                
                for container in update_containers:
                    try:
                        try:
                            update_number = container.find_element(By.CSS_SELECTOR, "span.type-13.soft-black_50.text-uppercase").text.strip()
                        except:
                            continue  
                        
                        if update_number in scraped_updates:
                            logger.info(f"Skipping already scraped update: {update_number}")
                            continue
                        
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

                        try:
                            read_more = container.find_element(By.CSS_SELECTOR, "button.ksr-button.bttn")
                            
                            logger.info(f"Clicking Read more button for {update_number}: {title}")
                            browser.execute_script("arguments[0].click();", read_more)
                            random_sleep(2, 3)
                            
                            WebDriverWait(browser, 10).until(
                                lambda x: x.execute_script("return document.readyState") == "complete"
                            )
                            
                            try:
                                content_div = browser.find_element(By.CSS_SELECTOR, "div.rte__content")
                                content = content_div.text.strip()
                                likes_button = browser.find_element(By.CSS_SELECTOR, "button.type-12.soft-black_50.text-underline")
                                likes_count = int(likes_button.text.split()[0])
                                
                                comments_section = browser.find_element(By.ID, "comments")
                                
                                comment_groups = comments_section.find_elements(By.CSS_SELECTOR, "div.w100p")
                                metadata_groups = comments_section.find_elements(By.CSS_SELECTOR, "div.flex.mb3.justify-between")
                                comments_data = []
                                
                                for i in range(len(metadata_groups)):
                                    metadata = metadata_groups[i]
                                    comment_group = comment_groups[i] if i < len(comment_groups) else None
                                    
                                    try:
                                        inner = metadata.find_element(By.CSS_SELECTOR, "div.flex")
                                        inner2 = inner.find_element(By.CSS_SELECTOR, "span.mr2")
                                        name_span = inner2.find_element(By.CSS_SELECTOR, "span.do-not-visually-track")
                                        commenter_name = name_span.text.strip()
                                    except Exception as e:
                                        logger.error(f"Error getting commenter name: {str(e)}")
                                        commenter_name = "Unknown"

                                    try:
                                        time_element = metadata.find_element(By.CSS_SELECTOR, "time[datetime]")
                                        timestamp = time_element.get_attribute("title") 
                                        datetime_value = time_element.get_attribute("datetime") 
                                        relative_time = time_element.text.strip() 
                                    except Exception as e:
                                        logger.error(f"Error getting timestamp: {str(e)}")
                                        timestamp = "Unknown"
                                        datetime_value = "Unknown"
                                        relative_time = "Unknown"
                                    
                                    full_text = ""
                                    if comment_group:
                                        try:
                                            comment_texts = comment_group.find_elements(By.CSS_SELECTOR, "p.data-comment-text.type-14")
                                            full_text = " ".join([text.text.strip() for text in comment_texts if text.text.strip()])
                                        except Exception as e:
                                            logger.error(f"Error getting comment text: {str(e)}")
                                    
                                    if full_text or commenter_name != "Unknown":  
                                        comment_data = {
                                            'text': full_text,
                                            'commenter': commenter_name,
                                            'timestamp': timestamp,
                                            'datetime': datetime_value,
                                            'relative_time': relative_time
                                        }
                                        comments_data.append(comment_data)
                                
                                comments_count = len(comments_data)
                                logger.info(f"Found {comments_count} comments")
                                
                                update_data = {
                                    'update_number': update_number,
                                    'title': title,
                                    'creator': creator,
                                    'date': date,
                                    'content': content,
                                    'url': browser.current_url,
                                    'likes_count': likes_count,
                                    'comments_count': comments_count,
                                    'comments': comments_data
                                }
                                updates.append(update_data)
                                scraped_updates.add(update_number)
                                logger.info(f"Successfully extracted update: {title[:50]}...")
                                
                                browser.back()
                                random_sleep(2, 3)
                                
                                WebDriverWait(browser, 10).until(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.bg-grey-100.pt6"))
                                )
                            except Exception as e:
                                logger.error(f"Error finding content: {str(e)}")
                                content = ""
                        except Exception as e:
                            logger.error(f"Error with read more button or content extraction: {str(e)}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error processing update container: {str(e)}")
                        continue
                
                if len(updates) < updates_count:
                    logger.info("Scrolling to load more updates...")
                    human_like_scroll(browser)
                    random_sleep(2, 3)
                
            except Exception as e:
                logger.error(f"Error in update processing loop: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in extract_updates_content: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    logger.info(f"Finished processing updates. Found {len(updates)} updates.")
    return updates 