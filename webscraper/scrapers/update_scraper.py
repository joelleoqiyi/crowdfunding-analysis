from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import traceback
from utils.browser_utils import random_sleep, human_like_scroll
from datetime import datetime

logger = logging.getLogger(__name__)

def extract_campaign_details(browser):
    """Extract campaign funding details from the main project page"""
    campaign_details = {}
    
    try:
        logger.info("Extracting campaign funding details...")
        WebDriverWait(browser, 10).until(
            lambda x: x.execute_script("return document.readyState") == "complete"
        )
        
        # Extract funding period
        try:
            # Based on the first image, targeting the funding period section with class "f5"
            funding_period_selectors = [
                "div.NS_campaigns__funding_period", 
                "div.funding_period",
                "p.f5",  # From the first image
                "div.f5",  # Alternative
                "div.flex.gap1", # New layout
                "p.campaign-state-description",
                "div.campaign-state-subtext"
            ]
            
            for selector in funding_period_selectors:
                try:
                    funding_period_container = browser.find_element(By.CSS_SELECTOR, selector)
                    funding_period_text = funding_period_container.text.strip()
                    if funding_period_text and ("—" in funding_period_text or "-" in funding_period_text or 
                                              funding_period_text.count("20") >= 2 or 
                                              any(month in funding_period_text.lower() for month in 
                                                  ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])):
                        campaign_details['funding_period'] = funding_period_text
                        logger.info(f"Extracted funding period: {funding_period_text}")
                        
                        # Extract start and end dates
                        time_elements = funding_period_container.find_elements(By.CSS_SELECTOR, "time[datetime]")
                        if len(time_elements) >= 2:
                            start_date = time_elements[0].get_attribute("datetime")
                            end_date = time_elements[1].get_attribute("datetime")
                            campaign_details['funding_start_date'] = start_date
                            campaign_details['funding_end_date'] = end_date
                            
                        # Extract duration if available (e.g., "30 days" from the image)
                        if "days" in funding_period_text:
                            try:
                                # Try to extract the number before "days"
                                parts = funding_period_text.split("days")
                                if len(parts) > 1:
                                    duration_str = parts[0].split("(")[-1].strip() if "(" in parts[0] else parts[0].strip()
                                    duration = int(''.join(filter(str.isdigit, duration_str)))
                                    campaign_details['funding_duration_days'] = duration
                                    logger.info(f"Extracted duration: {duration} days")
                            except Exception as e:
                                logger.error(f"Error extracting duration: {str(e)}")
                        
                        # Try to calculate duration from dates
                        if 'funding_start_date' in campaign_details and 'funding_end_date' in campaign_details and 'funding_duration_days' not in campaign_details:
                            try:
                                start_date = datetime.fromisoformat(campaign_details['funding_start_date'].replace('Z', '+00:00'))
                                end_date = datetime.fromisoformat(campaign_details['funding_end_date'].replace('Z', '+00:00'))
                                duration = (end_date - start_date).days
                                campaign_details['funding_duration_days'] = duration
                                logger.info(f"Calculated duration: {duration} days")
                            except Exception as e:
                                logger.error(f"Error calculating duration from dates: {str(e)}")
                        
                        break
                except Exception as e:
                    logger.error(f"Error with funding_period selector {selector}: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error extracting funding period: {str(e)}")
        
        # Try to get days remaining or funding period if not already found
        days_remaining = None
        time_selectors = [
            "div.ksr_page_timer",
            "div.campaign-state-header span.count",
            "div.NS_campaigns__spotlight_stats span.count",
            "div[data-test-id='campaign-time-remaining']",
            "div.mb4", # New layout
            "span.block.type-16.type-28-md"
        ]
        
        for selector in time_selectors:
            try:
                element = browser.find_element(By.CSS_SELECTOR, selector)
                time_text = element.text.strip().lower()
                if not time_text:
                    continue
                
                if 'days to go' in time_text or 'left' in time_text or 'to go' in time_text:
                    # Active campaign - days remaining
                    days_remaining = int(''.join(filter(str.isdigit, time_text)))
                    campaign_details['days_remaining'] = days_remaining
                    logger.info(f"Found days remaining: {days_remaining}")
                    break
                elif 'funded' in time_text or 'successful' in time_text:
                    # Successful campaign that's ended
                    days_remaining = 0
                    campaign_details['days_remaining'] = days_remaining
                    logger.info("Campaign has ended successfully")
                    break
                elif 'unsuccessful' in time_text or 'canceled' in time_text:
                    # Failed campaign that's ended
                    days_remaining = 0
                    campaign_details['days_remaining'] = days_remaining
                    logger.info("Campaign has ended unsuccessfully")
                    break
            except Exception as e:
                logger.error(f"Error with time selector {selector}: {str(e)}")
                continue
        
        # Extract funding goal and pledged amount
        # Try to get funding goal
        funding_goal = None
        goal_selectors = [
            "span.money.goal",
            "div.goal",
            "span[data-test-id='funding-goal']",
            "span[data-test-id='amount-goal']", 
            "div.mb3 > span.mb1",  # New layout
            "div.type-12.medium.navy-500 span.money",
            "div.navy-500 span.money",
            "span.block.dark-grey-500.type-12.type-14-md.lh3-lg span.money",
            "span.block.dark-grey-500.type-12.type-14-md.lh3-lg span.hide"
        ]
        
        for selector in goal_selectors:
            try:
                elements = browser.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    goal_text = element.text.strip()
                    if goal_text and ('$' in goal_text or '€' in goal_text or '£' in goal_text):
                        funding_goal = goal_text
                        logger.info(f"Found funding goal: {funding_goal}")
                        campaign_details['funding_goal'] = funding_goal
                        break
                if funding_goal:
                    break
            except Exception as e:
                logger.error(f"Error with goal selector {selector}: {str(e)}")
                continue
        
        # If still not found, look for text containing "goal"
        if not funding_goal:
            try:
                elements = browser.find_elements(By.XPATH, "//*[contains(text(), 'goal')]")
                for element in elements:
                    text = element.text.strip()
                    if 'goal' in text.lower() and any(c in text for c in ['$', '€', '£']):
                        # Try to extract the money value using regex
                        import re
                        goal_match = re.search(r'[\$\€\£][\d,]+(?:\.\d+)?', text)
                        if goal_match:
                            funding_goal = goal_match.group(0)
                            campaign_details['funding_goal'] = funding_goal
                            logger.info(f"Extracted funding goal via text: {funding_goal}")
                            break
            except Exception as e:
                logger.error(f"Error extracting goal via text: {str(e)}")
        
        # Try to get pledged amount
        pledged_amount = None
        pledged_selectors = [
            "span.ksr-green-700",
            "span.money.pledged",
            "div.pledged",
            "span[data-test-id='amount-pledged']",
            "div.type-16 > span.mb1", # New layout
            "span.ksr-green-700.inline-block.bold.type-16.type-28-md span.soft-black",
            "h3.mb0 span.money",
            "div.type-16 span"
        ]
        
        for selector in pledged_selectors:
            try:
                elements = browser.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    amount_text = element.text.strip()
                    if amount_text and ('$' in amount_text or '€' in amount_text or '£' in amount_text):
                        pledged_amount = amount_text
                        logger.info(f"Found pledged amount: {pledged_amount}")
                        campaign_details['pledged_amount'] = pledged_amount
                        break
                if pledged_amount:
                    break
            except Exception as e:
                logger.error(f"Error with pledged selector {selector}: {str(e)}")
                continue
        
        # Try to get backers count
        backers_count = None
        backers_selectors = [
            "div.backers-count",
            "span.ksr-green-700 + div",
            "div.NS_campaigns__spotlight_stats div.backers-wrap span",
            "div[data-test-id='backers-count']",
            "div.type-16 > div", # New layout
            "h3.mb0",
            "div.mb0 h3",
            "div.block.type-16 span"
        ]
        
        for selector in backers_selectors:
            try:
                elements = browser.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    count_text = element.text.strip()
                    if count_text and any(c.isdigit() for c in count_text):
                        # Make sure we're not grabbing other numbers like the funding amount
                        if not any(c in count_text for c in ['$', '€', '£']):
                            # Extract just the number
                            backers_count = int(''.join(filter(str.isdigit, count_text)))
                            logger.info(f"Found backers count: {backers_count}")
                            campaign_details['backers_count'] = backers_count
                            break
                if backers_count is not None:
                    break
            except Exception as e:
                logger.error(f"Error with backers selector {selector}: {str(e)}")
                continue
        
    except Exception as e:
        logger.error(f"Error in extract_campaign_details: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    return campaign_details

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
        
        # Get the current URL to determine if we need to navigate to updates page
        current_url = browser.current_url
        
        # Check if we're already on the updates page
        if "/posts" not in current_url:
            # We need to navigate to the updates page
            logger.info("Not on updates page, need to navigate there")
            # Properly construct the updates URL
            base_url = current_url.rstrip('/')
            updates_url = f"{base_url}/posts"
            logger.info(f"Navigating to updates URL: {updates_url}")
            browser.get(updates_url)
            random_sleep(3, 5)
            
            # Wait for page to load after navigation
            WebDriverWait(browser, 15).until(
                lambda x: x.execute_script("return document.readyState") == "complete"
            )

        try:
            logger.info("Looking for updates count...")
            # Try multiple selectors for finding the updates count
            count_selectors = [
                "a[href$='/posts'] span",
                "span.count",
                "li.block-menu-item--count span",
                "a[data-content='updates'] span",
                "div[data-test-id='updates-count']",
                "div.updates-count"
            ]
            
            updates_count = 0
            for selector in count_selectors:
                try:
                    count_element = browser.find_element(By.CSS_SELECTOR, selector)
                    count_text = count_element.text.strip()
                    if count_text and any(c.isdigit() for c in count_text):
                        updates_count = int(''.join(filter(str.isdigit, count_text)))
                        logger.info(f"Found updates count: {updates_count}")
                        break
                except Exception:
                    continue
            
            if updates_count == 0:
                logger.info("No updates found or couldn't determine update count")
                return {"updates": updates}
            
        except Exception as e:
            logger.error(f"Error finding updates count: {str(e)}")
            return {"updates": updates}

        max_attempts = updates_count + 2 
        attempt = 0
        
        while len(updates) < updates_count and attempt < max_attempts:
            attempt += 1
            logger.info(f"\nAttempt {attempt} to find updates. Found so far: {len(updates)}")
            
            try:
                # Try multiple selectors for the updates container
                container_selectors = [
                    "div.bg-grey-100.pt6",
                    "div.updates-content",
                    "div[data-test-id='updates-content']",
                    "div.grid-container.pb3"
                ]
                
                updates_container = None
                for selector in container_selectors:
                    try:
                        WebDriverWait(browser, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        updates_container = browser.find_element(By.CSS_SELECTOR, selector)
                        if updates_container:
                            break
                    except Exception:
                        continue
                
                if not updates_container:
                    logger.error("Could not find updates container with any selector")
                    break
                
                # Try multiple selectors for update items
                update_item_selectors = [
                    "div.grid-container",
                    "div.update-post",
                    "div[data-test-id='update-card']",
                    "div.mb3.pb3"
                ]
                
                update_containers = []
                for selector in update_item_selectors:
                    try:
                        containers = updates_container.find_elements(By.CSS_SELECTOR, selector)
                        if containers:
                            update_containers = containers
                            break
                    except Exception:
                        continue
                
                if not update_containers:
                    logger.info("No update containers found")
                    break
                
                for container in update_containers:
                    try:
                        try:
                            update_number = container.find_element(By.CSS_SELECTOR, "span.type-13.soft-black_50.text-uppercase, span.update-number").text.strip()
                        except:
                            continue  
                        
                        if update_number in scraped_updates:
                            logger.info(f"Skipping already scraped update: {update_number}")
                            continue
                        
                        try:
                            title = container.find_element(By.CSS_SELECTOR, "h2.kds-heading.mb3, h2.update-title").text.strip()
                        except:
                            title = "Unknown Title"
                        
                        try:
                            creator_div = container.find_element(By.CSS_SELECTOR, "div.pl2, div.creator-info")
                            creator = creator_div.text.replace("Creator", "").strip()
                        except:
                            creator = "Unknown"
                            
                        try:
                            date = container.find_element(By.CSS_SELECTOR, "span.type-13.soft-black_50.block-md, span.update-date").text.strip()
                        except:
                            date = None

                        try:
                            # Try multiple selectors for the Read More button
                            read_more_selectors = [
                                "button.ksr-button.bttn",
                                "button.read-more",
                                "button[data-test-id='read-more']",
                                "button.update-expand",
                                "button.bttn-sm.bttn-soft-black",
                                "button.bttn.bttn-sm.bttn-soft-black",
                                "button.bttn-medium"
                            ]
                            
                            read_more = None
                            for selector in read_more_selectors:
                                try:
                                    buttons = container.find_elements(By.CSS_SELECTOR, selector)
                                    for button in buttons:
                                        if button.is_displayed() and any(text in button.text.lower() for text in ['read more', 'show more', 'view more']):
                                            read_more = button
                                            break
                                    if read_more:
                                        break
                                except Exception:
                                    continue
                            
                            if not read_more:
                                logger.error("Could not find Read More button")
                                continue
                            
                            logger.info(f"Clicking Read more button for {update_number}: {title}")
                            browser.execute_script("arguments[0].click();", read_more)
                            random_sleep(2, 3)
                            
                            WebDriverWait(browser, 10).until(
                                lambda x: x.execute_script("return document.readyState") == "complete"
                            )
                            
                            try:
                                content_div = browser.find_element(By.CSS_SELECTOR, "div.rte__content, div.update-content")
                                content = content_div.text.strip()
                                
                                try:
                                    likes_button = browser.find_element(By.CSS_SELECTOR, "button.type-12.soft-black_50.text-underline, button.likes-count")
                                    likes_count = int(likes_button.text.split()[0])
                                except:
                                    likes_count = 0
                                
                                try:
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
                                except Exception as e:
                                    logger.error(f"Error processing comments: {str(e)}")
                                    comments_data = []
                                
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
                                
                                # Wait for updates page to reload
                                for selector in container_selectors:
                                    try:
                                        WebDriverWait(browser, 10).until(
                                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                                        )
                                        break
                                    except Exception:
                                        continue
                                
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
    return {"updates": updates} 
