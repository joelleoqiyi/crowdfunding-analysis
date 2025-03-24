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
                "div.f5"  # Alternative
            ]
            
            for selector in funding_period_selectors:
                try:
                    funding_period_container = browser.find_element(By.CSS_SELECTOR, selector)
                    funding_period_text = funding_period_container.text.strip()
                    if "—" in funding_period_text or "-" in funding_period_text or funding_period_text.count("20") >= 2:
                        campaign_details['funding_period_text'] = funding_period_text
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
                                except Exception as e:
                                    logger.error(f"Error extracting duration: {str(e)}")
                        break
                except:
                    continue
        except Exception as e:
            logger.error(f"Error extracting funding period: {str(e)}")
        
        # Extract funding amount, goal and backers
        try:
            # First try the successful project selectors
            money_selectors = [
                "span.money",
                "span.money-raised",
                "h3 span.money"  # From the second image
            ]
            
            # First extract pledged amount
            pledged_found = False
            for selector in money_selectors:
                try:
                    money_elements = browser.find_elements(By.CSS_SELECTOR, selector)
                    if len(money_elements) >= 1:
                        pledged_amount = money_elements[0].text.strip()
                        if pledged_amount:  # Only set if we actually got a value
                            campaign_details['pledged_amount'] = pledged_amount
                            logger.info(f"Extracted pledged amount: {pledged_amount}")
                            pledged_found = True
                            break
                except:
                    continue
            
            # If pledged amount not found, try failed project selector
            if not pledged_found:
                try:
                    logger.info("Starting failed project pledged amount extraction...")
                    # Try to find the outer flex container first
                    initial_goal_text = browser.find_element(By.CSS_SELECTOR, "div.grid-row.order2-md.hide-lg")
                    next = initial_goal_text.find_element(By.CSS_SELECTOR, "div.grid-col-12.grid-col-10-md.grid-col-offset-1-md")
                    next2 = next.find_element(By.CSS_SELECTOR, "div.flex.flex-column-lg.mb4.mb5-sm")
                    next3 = next2.find_element(By.CSS_SELECTOR, "div.mb4-lg")
                    pledged_amount1 = next3.find_element(By.CSS_SELECTOR, "span.ksr-green-700.inline-block.bold.type-16.type-28-md")
                    pledged_amount_element = pledged_amount1.find_element(By.CSS_SELECTOR, "span.soft-black")
                    
                    if pledged_amount_element:
                        pledged_amount = pledged_amount_element.text.strip()
                        if pledged_amount:
                            campaign_details['pledged_amount'] = pledged_amount
                            logger.info(f"Successfully extracted pledged amount (failed project): {pledged_amount}")
                            pledged_found = True
                        else:
                            logger.error("Found pledged amount element but text is empty")
                    else:
                        logger.error("Could not find pledged amount element")
                        
                except Exception as e:
                    logger.error(f"Error extracting failed project pledged amount: {str(e)}")
                    
                # If we still haven't found the amount, try the direct selector as last resort
                if not pledged_found:
                    try:
                        logger.info("Trying direct selector as last resort...")
                        soft_black_spans = browser.find_elements(By.CSS_SELECTOR, "span.ksr-green-700 span.soft-black")
                        if soft_black_spans:
                            pledged_amount = soft_black_spans[0].text.strip()
                            if pledged_amount:
                                campaign_details['pledged_amount'] = pledged_amount
                                logger.info(f"Successfully extracted pledged amount (direct selector): {pledged_amount}")
                                pledged_found = True
                            else:
                                logger.error("Found direct selector element but text is empty")
                        else:
                            logger.error("No pledged amount elements found with direct selector")
                    except Exception as e:
                        logger.error(f"Error with direct selector attempt: {str(e)}")
            
            if not pledged_found:
                logger.error("Failed to extract pledged amount using any method")
            
            # Now look for funding goal
            goal_found = False
            goal_selectors = [
                "div.type-12.medium.navy-500",
                "div.navy-500",
                "div[class*='navy-500']"
            ]
            
            for selector in goal_selectors:
                try:
                    goal_divs = browser.find_elements(By.CSS_SELECTOR, selector)
                    for div in goal_divs:
                        div_text = div.text.strip()
                        if "goal" in div_text.lower():
                            # Extract the money value
                            goal_spans = div.find_elements(By.CSS_SELECTOR, "span.money")
                            if goal_spans:
                                funding_goal = goal_spans[0].text.strip()
                                campaign_details['funding_goal'] = funding_goal
                                logger.info(f"Extracted funding goal: {funding_goal}")
                                goal_found = True
                                break
                            else:
                                # Try to extract using regex if no span.money found
                                import re
                                goal_match = re.search(r'\$[\d,]+', div_text)
                                if goal_match:
                                    funding_goal = goal_match.group(0)
                                    campaign_details['funding_goal'] = funding_goal
                                    logger.info(f"Extracted funding goal: {funding_goal}")
                                    goal_found = True
                                    break
                    if goal_found:
                        break
                except:
                    continue
            
            # If goal not found, try failed project selector
            if not goal_found:
                try:
                    logger.info("Starting failed project goal extraction...")
                    # Look for text "pledged of X goal" in dark-grey-500
                    initial_goal_text = browser.find_element(By.CSS_SELECTOR, "div.grid-row.order2-md.hide-lg")
                    next = initial_goal_text.find_element(By.CSS_SELECTOR, "div.grid-col-12.grid-col-10-md.grid-col-offset-1-md")
                    next2 = next.find_element(By.CSS_SELECTOR, "div.flex.flex-column-lg.mb4.mb5-sm")
                    next3 = next2.find_element(By.CSS_SELECTOR, "div.mb4-lg")
                    goal_text1 = next3.find_element(By.CSS_SELECTOR, "span.block.dark-grey-500.type-12.type-14-md.lh3-lg")
                    goal_text = goal_text1.find_element(By.CSS_SELECTOR, "span.inline-block-sm.hide")
                    
                    if goal_text:
                        # Try to find the money span directly
                        try:
                            money_span = goal_text.find_element(By.CSS_SELECTOR, "span.money")
                            if money_span:
                                funding_goal = money_span.get_attribute('textContent').strip()
                                if funding_goal:
                                    campaign_details['funding_goal'] = funding_goal
                                    logger.info(f"Extracted funding goal (failed project): {funding_goal}")
                            else:
                                logger.error("Could not find money span")
                        except Exception as e:
                            logger.error(f"Error finding money span: {str(e)}")
                            # Try regex as fallback
                            import re
                            html_content = goal_text.get_attribute('outerHTML')
                            goal_match = re.search(r'[S$\$][\d,]+', html_content)
                            if goal_match:
                                funding_goal = goal_match.group(0)
                                campaign_details['funding_goal'] = funding_goal
                                logger.info(f"Extracted funding goal using regex: {funding_goal}")
                            else:
                                logger.error("Could not extract goal amount using regex")
                    else:
                        logger.error("goal_text element is None")
                except Exception as e:
                    logger.error(f"Error extracting failed project goal: {str(e)}")
            
            # Backers count - try both successful and failed project selectors
            backers_found = False
            backers_selectors = [
                "div.mb0 h3.mb0", 
                "h3.mb0", 
                "div.type-12 h3",
                "div.mb0 h3",  # From the second image
                "h3.mbo"  # Alternative spelling
            ]
            
            for selector in backers_selectors:
                try:
                    backers_element = browser.find_element(By.CSS_SELECTOR, selector)
                    backers_count = backers_element.text.strip()
                    if backers_count.isdigit() or (backers_count and backers_count[0].isdigit()):
                        campaign_details['backers_count'] = backers_count
                        logger.info(f"Extracted backers count: {backers_count}")
                        backers_found = True
                        break
                except:
                    continue
            
            # If backers not found, try failed project selector
            if not backers_found:
                try:
                    # Look for backers count in block type-16 element
                    backers_div = browser.find_element(By.CSS_SELECTOR, "div.block.type-16")
                    if backers_div:
                        backers_count = backers_div.find_element(By.CSS_SELECTOR, "span").text.strip()
                        if backers_count.isdigit():
                            campaign_details['backers_count'] = backers_count
                            logger.info(f"Extracted backers count (failed project): {backers_count}")
                except Exception as e:
                    logger.error(f"Error extracting failed project backers count: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error extracting funding details: {str(e)}")
            
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
        
        # Store the current URL (main project page)
        main_page_url = browser.current_url
        
        # Get updates URL
        base_url = browser.current_url.rstrip('/')
        updates_url = base_url + "/posts"
        logger.info(f"Updates URL: {updates_url}")
        
        try:
            logger.info("Looking for updates count...")
            # Try to find the updates count link
            updates_link = WebDriverWait(browser, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[href$='/posts']"))
            )
            
            # Try different methods to get the updates count
            updates_count = 0
            
            # Method 1: Try to find a span element inside the updates link
            try:
                span_elements = updates_link.find_elements(By.CSS_SELECTOR, "span")
                if span_elements:
                    for span in span_elements:
                        span_text = span.text.strip()
                        if span_text and any(c.isdigit() for c in span_text):
                            updates_count_text = span_text
                            updates_count = int(''.join(filter(str.isdigit, updates_count_text)))
                            logger.info(f"Found updates count from span: {updates_count}")
                            break
            except Exception as e:
                logger.warning(f"Could not find updates count from span: {str(e)}")
            
            # Method 2: Try to extract from the link text
            if updates_count == 0:
                try:
                    link_text = updates_link.text.strip()
                    if link_text and any(c.isdigit() for c in link_text):
                        updates_count = int(''.join(filter(str.isdigit, link_text)))
                        logger.info(f"Found updates count from link text: {updates_count}")
                except Exception as e:
                    logger.warning(f"Could not find updates count from link text: {str(e)}")
            
            # Method 3: Navigate to the updates page and count the updates
            if updates_count == 0:
                logger.info("Could not determine updates count from link, will check after navigation")
            
            # If there are no updates, return early
            if updates_count == 0:
                # We'll still navigate to the updates page to confirm
                logger.info("No updates found from link, will check the updates page directly")
            else:
                logger.info(f"Found updates count: {updates_count}")
                
        except Exception as e:
            logger.error(f"Error finding updates link: {str(e)}")
            return {"updates": []}

        # Try clicking the updates link directly first
        try:
            logger.info("Trying to click updates link directly...")
            updates_link.click()
            random_sleep(2, 3)
            # Wait for page to load after clicking
            WebDriverWait(browser, 10).until(
                lambda x: x.execute_script("return document.readyState") == "complete"
            )
        except Exception as e:
            logger.error(f"Error clicking updates link: {str(e)}")
            # If clicking fails, navigate using the URL
            logger.info(f"Navigating to updates page via URL: {updates_url}")
            browser.get(updates_url)
            random_sleep(3, 5)
            # Wait for page to load after navigation
            WebDriverWait(browser, 10).until(
                lambda x: x.execute_script("return document.readyState") == "complete"
            )
        
        # Verify we're on the updates page
        if "/posts" not in browser.current_url:
            logger.error(f"Failed to navigate to updates page. Current URL: {browser.current_url}")
            # Try one more time with the URL
            browser.get(updates_url)
            random_sleep(3, 5)
            # Wait for page to load after navigation
            WebDriverWait(browser, 10).until(
                lambda x: x.execute_script("return document.readyState") == "complete"
            )
            
            if "/posts" not in browser.current_url:
                logger.error("Still failed to navigate to updates page. Returning empty updates.")
                return {"updates": []}

        # If we couldn't determine the updates count from the link, try to count them now
        if updates_count == 0:
            try:
                # Wait for the updates container to load
                update_container_selectors = [
                    "div.bg-grey-100.pt6",
                    "div.posts-container",
                    "div.grid-container.pb10",
                    "div.container-flex",
                    "div.post-index",
                    "main.NS_posts__body"
                ]
                
                updates_container = None
                for selector in update_container_selectors:
                    try:
                        WebDriverWait(browser, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        updates_container = browser.find_element(By.CSS_SELECTOR, selector)
                        break
                    except:
                        continue
                
                if updates_container:
                    # Try to find update items
                    update_item_selectors = [
                        "div.grid-container",
                        "div.post-card",
                        "div.post-container",
                        "div.post",
                        "article.post"
                    ]
                    
                    for selector in update_item_selectors:
                        try:
                            items = updates_container.find_elements(By.CSS_SELECTOR, selector)
                            if items:
                                updates_count = len(items)
                                logger.info(f"Found {updates_count} updates by counting items on the page")
                                break
                        except:
                            continue
                
                # If we still couldn't find any updates, check if there's a "no updates" message
                if updates_count == 0:
                    page_text = browser.page_source.lower()
                    if "no updates" in page_text or "hasn't posted any updates" in page_text:
                        logger.info("Confirmed no updates for this project")
                        return {"updates": []}
            except Exception as e:
                logger.error(f"Error counting updates on the page: {str(e)}")
                # If we can't determine the count, assume there are no updates
                return {"updates": []}

        # Process updates until we've found them all or hit the attempt limit
        max_attempts = updates_count + 2 if updates_count > 0 else 2
        attempt = 0
        consecutive_read_more_failures = 0  # Track consecutive failures to find read more button
        
        while len(updates) < updates_count and attempt < max_attempts:
            attempt += 1
            logger.info(f"\nAttempt {attempt} to find updates. Found so far: {len(updates)}")
            
            try:
                # Wait for the updates container to load
                # Try multiple selectors for the updates container
                update_container_selectors = [
                    "div.bg-grey-100.pt6",
                    "div.posts-container",
                    "div.grid-container.pb10",
                    "div.container-flex",
                    "div.post-index",  # Additional selector
                    "main.NS_posts__body"  # Additional selector
                ]
                
                updates_container = None
                for selector in update_container_selectors:
                    try:
                        logger.info(f"Trying to find updates container with selector: {selector}")
                        WebDriverWait(browser, 5).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                        )
                        updates_container = browser.find_element(By.CSS_SELECTOR, selector)
                        logger.info(f"Found updates container with selector: {selector}")
                        break
                    except:
                        continue
                
                if not updates_container:
                    logger.error("Could not find updates container with any selector")
                    consecutive_read_more_failures += 1
                    if consecutive_read_more_failures >= 2:
                        logger.info("Failed to find updates container twice, breaking loop")
                        break
                    continue
                
                # Try multiple selectors for update items
                update_item_selectors = [
                    "div.grid-container",
                    "div.post-card",
                    "div.post-container",
                    "div.post",  # Additional selector
                    "article.post"  # Additional selector
                ]
                
                update_containers = []
                for selector in update_item_selectors:
                    try:
                        items = updates_container.find_elements(By.CSS_SELECTOR, selector)
                        if items:
                            update_containers = items
                            logger.info(f"Found {len(items)} update items with selector: {selector}")
                            break
                    except:
                        continue
                
                if not update_containers:
                    logger.info("No update containers found")
                    # Try a different approach - look for update titles
                    try:
                        update_titles = browser.find_elements(By.CSS_SELECTOR, "h2.kds-heading, h2.post-title")
                        if update_titles:
                            logger.info(f"Found {len(update_titles)} update titles as fallback")
                            # For each title, try to find its parent container
                            for title in update_titles:
                                try:
                                    # Go up a few levels to find the container
                                    container = title
                                    for _ in range(3):  # Try going up 3 levels
                                        container = container.find_element(By.XPATH, "..")
                                        if container.tag_name in ['div', 'article']:
                                            update_containers.append(container)
                                            break
                                except:
                                    continue
                            logger.info(f"Found {len(update_containers)} update containers from titles")
                        else:
                            break
                    except Exception as e:
                        logger.error(f"Error in fallback update container search: {str(e)}")
                        break
                
                read_more_found = False  # Track if any read more button was found in this attempt
                
                for container in update_containers:
                    try:
                        try:
                            # Try multiple selectors for update number
                            update_number_selectors = [
                                "span.type-13.soft-black_50.text-uppercase",
                                "span.post-number",
                                "span.update-number",
                                "div.post-number"
                            ]
                            
                            update_number = None
                            for selector in update_number_selectors:
                                try:
                                    element = container.find_element(By.CSS_SELECTOR, selector)
                                    update_number = element.text.strip()
                                    if update_number and ("UPDATE" in update_number.upper() or "#" in update_number):
                                        logger.info(f"Found update number: {update_number}")
                                        break
                                except:
                                    continue
                                    
                            if not update_number:
                                logger.info("Could not find update number, generating a placeholder")
                                update_number = f"UPDATE #{len(updates) + 1}"
                        except:
                            continue  
                        
                        if update_number in scraped_updates:
                            logger.info(f"Skipping already scraped update: {update_number}")
                            continue
                        
                        try:
                            # Try multiple selectors for title
                            title_selectors = [
                                "h2.kds-heading.mb3",
                                "h2.post-title",
                                "h2.title",
                                "h2"
                            ]
                            
                            title = "Unknown Title"
                            for selector in title_selectors:
                                try:
                                    element = container.find_element(By.CSS_SELECTOR, selector)
                                    title_text = element.text.strip()
                                    if title_text:
                                        title = title_text
                                        logger.info(f"Found title: {title}")
                                        break
                                except:
                                    continue
                        except:
                            title = "Unknown Title"
                        
                        try:
                            # Try multiple selectors for creator
                            creator_selectors = [
                                "div.pl2",
                                "div.creator",
                                "span.creator-name",
                                "div.post-author"
                            ]
                            
                            creator = "Unknown"
                            for selector in creator_selectors:
                                try:
                                    element = container.find_element(By.CSS_SELECTOR, selector)
                                    creator_text = element.text.strip()
                                    if creator_text:
                                        creator = creator_text.replace("Creator", "").strip()
                                        logger.info(f"Found creator: {creator}")
                                        break
                                except:
                                    continue
                        except:
                            creator = "Unknown"
                            
                        try:
                            # Try multiple selectors for date
                            date_selectors = [
                                "span.type-13.soft-black_50.block-md",
                                "span.post-date",
                                "time.post-date",
                                "div.post-date"
                            ]
                            
                            date = None
                            for selector in date_selectors:
                                try:
                                    element = container.find_element(By.CSS_SELECTOR, selector)
                                    date_text = element.text.strip()
                                    if date_text:
                                        date = date_text
                                        logger.info(f"Found date: {date}")
                                        break
                                except:
                                    continue
                        except:
                            date = None

                        try:
                            # Try multiple selectors for read more button
                            read_more_selectors = [
                                "button.ksr-button.bttn",
                                "a.read-more",
                                "a.view-update",
                                "a[href*='/posts/']"
                            ]
                            
                            read_more = None
                            for selector in read_more_selectors:
                                try:
                                    elements = container.find_elements(By.CSS_SELECTOR, selector)
                                    for element in elements:
                                        if element.is_displayed() and (
                                            "read" in element.text.lower() or 
                                            "more" in element.text.lower() or 
                                            "view" in element.text.lower() or
                                            "/posts/" in element.get_attribute("href")
                                        ):
                                            read_more = element
                                            read_more_found = True  # Mark that we found at least one read more button
                                            break
                                    if read_more:
                                        break
                                except:
                                    continue
                            
                            if not read_more:
                                logger.error("Could not find read more button")
                                continue
                                
                            logger.info(f"Clicking Read more button for {update_number}: {title}")
                            # Store the current URL before clicking
                            updates_page_url = browser.current_url
                            
                            # Try to click the button
                            try:
                                browser.execute_script("arguments[0].click();", read_more)
                            except:
                                # If JavaScript click fails, try regular click
                                read_more.click()
                                
                            random_sleep(2, 3)
                            
                            # Wait for the update page to load
                            WebDriverWait(browser, 10).until(
                                lambda x: x.execute_script("return document.readyState") == "complete"
                            )
                            
                            try:
                                # Try multiple selectors for content
                                content_selectors = [
                                    "div.rte__content",
                                    "div.post-body",
                                    "div.post-content",
                                    "article.post"
                                ]
                                
                                content = ""
                                for selector in content_selectors:
                                    try:
                                        content_div = browser.find_element(By.CSS_SELECTOR, selector)
                                        content_text = content_div.text.strip()
                                        if content_text:
                                            content = content_text
                                            logger.info(f"Found content with selector: {selector}")
                                            break
                                    except:
                                        continue
                                
                                # Try multiple selectors for likes button
                                likes_selectors = [
                                    "button.type-12.soft-black_50.text-underline",
                                    "button.like-button",
                                    "div.like-count",
                                    "span.like-count"
                                ]
                                
                                likes_count = 0
                                for selector in likes_selectors:
                                    try:
                                        likes_button = browser.find_element(By.CSS_SELECTOR, selector)
                                        likes_text = likes_button.text.strip()
                                        if likes_text:
                                            likes_count = int(''.join(filter(str.isdigit, likes_text)))
                                            logger.info(f"Found likes count: {likes_count}")
                                            break
                                    except:
                                        continue
                                
                                # Try multiple selectors for comments section
                                comments_selectors = [
                                    "div#comments",
                                    "div.comments-container",
                                    "section.comments"
                                ]
                                
                                comments_section = None
                                for selector in comments_selectors:
                                    try:
                                        section = browser.find_element(By.CSS_SELECTOR, selector)
                                        if section.is_displayed():
                                            comments_section = section
                                            logger.info(f"Found comments section with selector: {selector}")
                                            break
                                    except:
                                        continue
                                
                                comments_data = []
                                if comments_section:
                                    # Process comments as before
                                    comment_groups = comments_section.find_elements(By.CSS_SELECTOR, "div.w100p")
                                    metadata_groups = comments_section.find_elements(By.CSS_SELECTOR, "div.flex.mb3.justify-between")
                                    
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
                                
                                # Navigate back to the updates page
                                logger.info(f"Navigating back to updates page: {updates_page_url}")
                                browser.get(updates_page_url)
                                random_sleep(2, 3)
                                
                                # Wait for the updates page to load
                                WebDriverWait(browser, 10).until(
                                    lambda x: x.execute_script("return document.readyState") == "complete"
                                )
                                
                                # Verify we're back on the updates page
                                if "/posts" not in browser.current_url:
                                    logger.error(f"Failed to navigate back to updates page. Current URL: {browser.current_url}")
                                    browser.get(updates_page_url)  # Try again
                                    random_sleep(2, 3)
                                    WebDriverWait(browser, 10).until(
                                        lambda x: x.execute_script("return document.readyState") == "complete"
                                    )
                            except Exception as e:
                                logger.error(f"Error finding content: {str(e)}")
                                content = ""
                                # Navigate back to the updates page
                                browser.get(updates_page_url)
                                random_sleep(2, 3)
                        except Exception as e:
                            logger.error(f"Error with read more button or content extraction: {str(e)}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error processing update container: {str(e)}")
                        continue
                
                if not read_more_found:
                    consecutive_read_more_failures += 1
                    if consecutive_read_more_failures >= 2:
                        logger.info("Failed to find any read more buttons twice, breaking loop")
                        break
                else:
                    consecutive_read_more_failures = 0  # Reset counter if we found any read more button
                
                if len(updates) < updates_count:
                    logger.info("Scrolling to load more updates...")
                    human_like_scroll(browser)
                    random_sleep(2, 3)
                
            except Exception as e:
                logger.error(f"Error in update processing loop: {str(e)}")
                consecutive_read_more_failures += 1
                if consecutive_read_more_failures >= 2:
                    logger.info("Failed to process updates twice, breaking loop")
                    break
                continue

    except Exception as e:
        logger.error(f"Error in extract_updates_content: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    logger.info(f"Finished processing updates. Found {len(updates)} updates.")
    return {"updates": updates} 