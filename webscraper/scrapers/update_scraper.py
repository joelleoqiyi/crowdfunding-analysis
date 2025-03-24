from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import traceback
from utils.browser_utils import random_sleep, human_like_scroll
from datetime import datetime, timedelta

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
                "span.block.navy-600.type-12.type-14-md.lh3-lg", # For live projects
                "div.type-12.type-14-md" # Another alternative for live projects
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
        
        # Extract days left for live projects
        try:
            # Look for the days left indicator shown in the image (26 days left)
            days_left_found = False
            
            # Try multiple selectors for the days left indicator
            days_left_selectors = [
                "span.block.type-16.type-28-md.bold.dark-grey-500",  # From the provided image
                "span.block.dark-grey-500", # Alternative 
                "div.ml5.ml0-lg div span", # Another structure
                "div.days-left span", # Generic selector
                "span[class*='dark-grey-500']" # Fallback using partial class match
            ]
            
            for selector in days_left_selectors:
                try:
                    days_left_elements = browser.find_elements(By.CSS_SELECTOR, selector)
                    
                    for element in days_left_elements:
                        days_text = element.text.strip()
                        # Check if it's just a number (like "26")
                        if days_text.isdigit():
                            # Try to find if "days" text is nearby
                            try:
                                # Check parent element for "days" text
                                parent = element.find_element(By.XPATH, "..")
                                parent_text = parent.text.lower()
                                
                                # Look for next sibling element that might contain "days left" text
                                siblings = parent.find_elements(By.XPATH, "./following-sibling::*")
                                
                                if "day" in parent_text or "days" in parent_text or any("day" in sib.text.lower() for sib in siblings):
                                    days_left = int(days_text)
                                    campaign_details['days_left'] = days_left
                                    logger.info(f"Extracted days left: {days_left}")
                                    days_left_found = True
                                    
                                    # If we found days left but don't have funding end date, calculate it
                                    if 'funding_end_date' not in campaign_details:
                                        end_date = datetime.now() + timedelta(days=days_left)
                                        campaign_details['funding_end_date'] = end_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                                        logger.info(f"Calculated funding end date from days left: {campaign_details['funding_end_date']}")
                                    
                                    # Look for funding period text nearby for live projects
                                    if 'funding_period_text' not in campaign_details or 'funding_start_date' not in campaign_details:
                                        try:
                                            # Look for nearby elements that might contain start date information
                                            funding_spans = parent.find_elements(By.XPATH, "./preceding-sibling::*[contains(@class, 'navy') or contains(@class, 'grey')]")
                                            funding_spans.extend(parent.find_elements(By.XPATH, "./following-sibling::*[contains(@class, 'navy') or contains(@class, 'grey')]"))
                                            
                                            for span in funding_spans:
                                                span_text = span.text.strip().lower()
                                                # Look for text indicating a date pattern (e.g., "launched" followed by a date)
                                                if ("launched" in span_text or "campaign started" in span_text or any(month.lower() in span_text for month in ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])):
                                                    # Store this text as a funding period indicator
                                                    campaign_details['live_funding_info'] = span_text
                                                    logger.info(f"Found live project funding info: {span_text}")
                                                    
                                                    # Try to extract any datetime elements
                                                    time_elements = span.find_elements(By.CSS_SELECTOR, "time[datetime]")
                                                    if time_elements:
                                                        start_date = time_elements[0].get_attribute("datetime")
                                                        if start_date:
                                                            campaign_details['funding_start_date'] = start_date
                                                            logger.info(f"Extracted funding start date for live project: {start_date}")
                                                            break
                                        except Exception as e:
                                            logger.error(f"Error extracting funding start date for live project: {str(e)}")
                                    
                                    break
                            except Exception as e:
                                logger.error(f"Error checking days left context: {str(e)}")
                            
                            # If we couldn't verify from the parent/siblings but it's a small number, assume it's days left
                            if not days_left_found and int(days_text) < 100:  # Assume no campaign runs for more than 100 days
                                days_left = int(days_text)
                                campaign_details['days_left'] = days_left
                                logger.info(f"Extracted likely days left: {days_left}")
                                days_left_found = True
                                break
                    
                    if days_left_found:
                        break
                        
                except Exception as e:
                    logger.error(f"Error with days left selector {selector}: {str(e)}")
                    continue
                    
            # Alternative approach: look for text containing "days left", "days to go", etc.
            if not days_left_found:
                try:
                    # Search for text patterns indicating days left
                    patterns = ["days left", "days to go", "days remaining"]
                    elements = browser.find_elements(By.XPATH, "//*[contains(text(), 'days left') or contains(text(), 'days to go') or contains(text(), 'days remaining')]")
                    
                    for element in elements:
                        text = element.text.strip().lower()
                        for pattern in patterns:
                            if pattern in text:
                                # Extract the number before the pattern
                                number_part = text.split(pattern)[0].strip()
                                if any(c.isdigit() for c in number_part):
                                    days_left = int(''.join(filter(str.isdigit, number_part)))
                                    campaign_details['days_left'] = days_left
                                    logger.info(f"Extracted days left from text: {days_left}")
                                    days_left_found = True
                                    break
                        
                        if days_left_found:
                            break
                except Exception as e:
                    logger.error(f"Error with text pattern search for days left: {str(e)}")
            
            # Try to extract start date from navy-600 element for live projects
            # This is specifically targeting the structure seen in the image
            if 'funding_start_date' not in campaign_details:
                try:
                    # Look for the navy-600 span that often shows launch date
                    navy_span_selectors = [
                        "span.block.navy-600.type-12.type-14-md.lh3-lg",
                        "span.navy-600", 
                        "span[class*='navy']"
                    ]
                    
                    for selector in navy_span_selectors:
                        try:
                            navy_spans = browser.find_elements(By.CSS_SELECTOR, selector)
                            for span in navy_spans:
                                span_text = span.text.strip()
                                # Check if it contains launch or campaign info
                                if span_text and ("launch" in span_text.lower() or any(month.lower() in span_text.lower() for month in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])):
                                    campaign_details['live_project_launch_text'] = span_text
                                    logger.info(f"Found live project launch info: {span_text}")
                                    
                                    # Try to extract a date using regex
                                    import re
                                    # Match patterns like "Jan 15, 2023" or "January 15, 2023" or "15 Jan 2023"
                                    date_patterns = [
                                        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4}',
                                        r'\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}'
                                    ]
                                    
                                    for pattern in date_patterns:
                                        date_match = re.search(pattern, span_text.lower())
                                        if date_match:
                                            date_str = date_match.group(0)
                                            campaign_details['extracted_start_date_text'] = date_str
                                            logger.info(f"Extracted potential start date text: {date_str}")
                                            break
                                    
                                    # Check for time elements that contain the datetime attribute
                                    time_elements = span.find_elements(By.CSS_SELECTOR, "time[datetime]")
                                    if time_elements:
                                        start_date = time_elements[0].get_attribute("datetime")
                                        if start_date:
                                            campaign_details['funding_start_date'] = start_date
                                            logger.info(f"Extracted funding start date from navy span: {start_date}")
                                            break
                            
                            # Break if we found the start date
                            if 'funding_start_date' in campaign_details:
                                break
                                
                        except Exception as e:
                            logger.error(f"Error with navy span selector {selector}: {str(e)}")
                            continue
                except Exception as e:
                    logger.error(f"Error extracting start date from navy spans: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error extracting days left: {str(e)}")
        
        # Extract funding amount, goal and backers
        try:
            # Based on the second image, targeting the money spans for pledged amount
            money_selectors = [
                "span.money",
                "span.money-raised",
                "h3 span.money",  # From the second image
                "span.soft-black",  # For live projects (Circular Ring 2 format)
                "h3.mb0",  # Alternative for pledged amount
                "h2.mb0",  # Alternative for pledged amount
                "span.ksr-green-700 span"  # Another alternative for live projects
            ]
            
            # First extract pledged amount
            pledged_found = False
            for selector in money_selectors:
                try:
                    money_elements = browser.find_elements(By.CSS_SELECTOR, selector)
                    if len(money_elements) >= 1:
                        pledged_amount = money_elements[0].text.strip()
                        if pledged_amount and any(c.isdigit() for c in pledged_amount):  # Only set if we actually got a value with numbers
                            campaign_details['pledged_amount'] = pledged_amount
                            logger.info(f"Extracted pledged amount: {pledged_amount}")
                            pledged_found = True
                            break
                except:
                    continue
                    
            # Try new format from first screenshot (Circular Ring 2)
            if not pledged_found:
                try:
                    logger.info("Trying to extract pledged amount for live project (new format)...")
                    # Direct selector for the S$ 1,949,328 format
                    pledged_elements = browser.find_elements(By.CSS_SELECTOR, "span.ksr-green-700, div.mb3 h2, div.mb3 h3, span.inline-block.bold, span.type-28-md, div.type-28-md")
                    
                    for element in pledged_elements:
                        pledged_text = element.text.strip()
                        if pledged_text and ('$' in pledged_text or '€' in pledged_text or '£' in pledged_text) and any(c.isdigit() for c in pledged_text):
                            campaign_details['pledged_amount'] = pledged_text
                            logger.info(f"Extracted pledged amount (live project): {pledged_text}")
                            pledged_found = True
                            break
                except Exception as e:
                    logger.error(f"Error with direct extraction of live project pledged amount: {str(e)}")
            
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
            
            # Goal extraction - add new selectors for live projects
            goal_found = False
            logger.info("Starting funding goal extraction with multiple methods...")
            
            # Method 1: Try standard selectors
            goal_selectors = [
                "div.type-12.medium.navy-500",
                "div.navy-500",
                "div[class*='navy-500']",
                "span.block.dark-grey-500",  # New selector for live projects
                "div.mb3 div",  # New selector for pledged of X goal format
                "span.inline-block-sm.hide span.money"  # Direct selector from the HTML image
            ]
            
            logger.info(f"Trying {len(goal_selectors)} different selectors for funding goal...")
            
            for selector in goal_selectors:
                try:
                    goal_divs = browser.find_elements(By.CSS_SELECTOR, selector)
                    
                    for i, div in enumerate(goal_divs):
                        try:
                            div_text = div.text.strip()
                            
                            # Check if this is the specific money span inside "pledged of" text
                            if selector == "span.inline-block-sm.hide span.money":
                                # This is our target selector from the image
                                funding_goal = div.text.strip()
                                if funding_goal:
                                    campaign_details['funding_goal'] = funding_goal
                                    logger.info(f"Extracted funding goal (direct selector): {funding_goal}")
                                    goal_found = True
                                    break
                            # Process other selectors as before
                            else:
                                if "goal" in div_text.lower():
                                    # Extract the money value
                                    goal_spans = div.find_elements(By.CSS_SELECTOR, "span.money")
                                    
                                    if goal_spans:
                                        # Bug fix: The first span might be empty, so we need to check all spans
                                        # and use the one that actually has content
                                        for span in goal_spans:
                                            span_text = span.text.strip()
                                            if span_text:
                                                funding_goal = span_text
                                                break
                                        else:
                                            # If no non-empty span was found, use the last one as fallback
                                            funding_goal = goal_spans[-1].text.strip()
                                            
                                        if funding_goal:  # Make sure we have a non-empty value
                                            campaign_details['funding_goal'] = funding_goal
                                            logger.info(f"Extracted funding goal: {funding_goal}")
                                            goal_found = True
                                            break
                                    else:
                                        # Try to extract using regex if no span.money found
                                        import re
                                        # Handle cases like S$, $, €, £
                                        goal_match = re.search(r'(S\$|\$|€|£)\s*[\d,]+', div_text)
                                        if goal_match:
                                            funding_goal = goal_match.group(0)
                                            campaign_details['funding_goal'] = funding_goal
                                            logger.info(f"Extracted funding goal: {funding_goal}")
                                            goal_found = True
                                            break
                            if 'funding_goal' in campaign_details:
                                break
                        except Exception as e:
                            logger.error(f"Error processing div with selector {selector}: {str(e)}")
                    if goal_found:
                        break
                except Exception as e:
                    logger.error(f"Error with selector {selector}: {str(e)}")
                    continue
            
            # Method 2: Add an additional attempt to find the exact element structure from the image
            if not goal_found:
                try:
                    logger.info("Trying exact element structure for funding goal from image...")
                    # Looking specifically for span.inline-block-sm.hide containing "pledged of" and a span.money
                    hide_spans = browser.find_elements(By.CSS_SELECTOR, "span.inline-block-sm.hide, span.hide")
                    
                    for i, span in enumerate(hide_spans):
                        try:
                            span_text = span.text.strip()
                            
                            # Check if this contains "pledged of" text
                            if "pledged of" in span_text.lower():
                                # Find the money span inside
                                try:
                                    money_spans = span.find_elements(By.CSS_SELECTOR, "span.money")
                                    
                                    if money_spans:
                                        money_span = money_spans[0]
                                        funding_goal = money_span.text.strip()
                                        
                                        if funding_goal:
                                            campaign_details['funding_goal'] = funding_goal
                                            logger.info(f"Extracted funding goal (exact structure): {funding_goal}")
                                            goal_found = True
                                            break
                                except Exception as e:
                                    logger.error(f"Error finding money span: {str(e)}")
                        except Exception as e:
                            continue
                except Exception as e:
                    logger.error(f"Error in exact structure extraction: {str(e)}")
                
            # Method 3: If goal still not found, try the parent element with the specific class
            if not goal_found:
                try:
                    logger.info("Trying to extract goal through parent span structure...")
                    # Find the specific dark-grey-500 span from the image
                    parent_spans = browser.find_elements(By.CSS_SELECTOR, "span.block.dark-grey-500.type-12.type-14-md.lh3-lg")
                    
                    for i, parent_span in enumerate(parent_spans):
                        try:
                            span_text = parent_span.text.strip()
                            
                            # Look for the text that contains both "pledged of" and "goal"
                            span_text_lower = span_text.lower()
                            if "pledged of" in span_text_lower:
                                
                                # Find all money spans inside
                                money_spans = parent_span.find_elements(By.CSS_SELECTOR, "span.money")
                                
                                if len(money_spans) > 1:  # There should be at least 2 (pledged and goal)
                                    # Find the non-empty money span that should be the goal
                                    funding_goal = ""
                                    # Try the second span first (most likely the goal)
                                    if money_spans[1].text.strip():
                                        funding_goal = money_spans[1].text.strip()
                                    else:
                                        # Otherwise check all spans and use the first non-empty one
                                        for span in money_spans:
                                            if span.text.strip():
                                                funding_goal = span.text.strip()
                                                break
                                    
                                    if funding_goal:
                                        campaign_details['funding_goal'] = funding_goal
                                        logger.info(f"Extracted funding goal (parent structure): {funding_goal}")
                                        goal_found = True
                                        break
                                else:
                                    # Extract using regex as fallback
                                    import re
                                    html = parent_span.get_attribute('innerHTML')
                                    
                                    # Find monetary values after "pledged of"
                                    pledged_of_index = html.lower().find("pledged of")
                                    if pledged_of_index > -1:
                                        substring = html[pledged_of_index:]
                                        
                                        # Handle cases like S$, $, €, £
                                        goal_match = re.search(r'(S\$|\$|€|£)\s*[\d,]+', substring)
                                        if goal_match:
                                            funding_goal = goal_match.group(0)
                                            campaign_details['funding_goal'] = funding_goal
                                            logger.info(f"Extracted funding goal (regex in parent): {funding_goal}")
                                            goal_found = True
                                            break
                        except Exception as e:
                            continue
                except Exception as e:
                    logger.error(f"Error in parent span extraction: {str(e)}")
            
            # Method 4: Final fallback - direct DOM traversal
            if not goal_found:
                try:
                    logger.info("Attempting direct DOM traversal as final fallback...")
                    # Try a more direct approach using JavaScript
                    js_script = """
                    var goalText = '';
                    var elements = document.querySelectorAll('*');
                    for (var i = 0; i < elements.length; i++) {
                        var el = elements[i];
                        if (el.textContent && el.textContent.toLowerCase().includes('goal') && 
                            /(S\\$|\\$|€|£)\\s*[\\d,]+/.test(el.textContent)) {
                            goalText = el.textContent;
                            break;
                        }
                    }
                    return goalText;
                    """
                    
                    goal_text = browser.execute_script(js_script)
                    
                    if goal_text:
                        # Extract the monetary value
                        import re
                        goal_match = re.search(r'(S\$|\$|€|£)\s*[\d,]+', goal_text)
                        if goal_match:
                            funding_goal = goal_match.group(0)
                            campaign_details['funding_goal'] = funding_goal
                            logger.info(f"Extracted funding goal (JS fallback): {funding_goal}")
                            goal_found = True
                except Exception as e:
                    logger.error(f"Error in JavaScript fallback: {str(e)}")
            
            if goal_found:
                logger.info(f"Successfully found funding goal: {campaign_details['funding_goal']}")
            else:
                logger.error("All funding goal extraction methods failed")
            
            # Backers count extraction
            backers_found = False
            backers_selectors = [
                "div.mb0 h3.mb0", 
                "h3.mb0", 
                "div.type-12 h3",
                "div.mb0 h3",  # From the second image
                "h3.mbo",  # Alternative spelling
                "div.backers h3",  # New selector
                "div.mb3 h3"  # New selector for live project format
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
            
            # Try direct extraction for the backers value
            if not backers_found:
                try:
                    logger.info("Trying to extract backers count for live project (new format)...")
                    # Look for numbers that could be backers (near "backers" text)
                    backers_candidates = browser.find_elements(By.CSS_SELECTOR, "h3.mb0, div.mb0 h3, div.type-16, div.mb3 h3")
                    
                    for element in backers_candidates:
                        text = element.text.strip()
                        if text.isdigit() or (text and text[0].isdigit()):
                            # Check if "backers" text is nearby
                            try:
                                parent = element.find_element(By.XPATH, "..")
                                parent_text = parent.text.lower()
                                if "backer" in parent_text or "backers" in parent_text:
                                    campaign_details['backers_count'] = text
                                    logger.info(f"Extracted backers count (live project direct): {text}")
                                    backers_found = True
                                    break
                            except:
                                # If we can't check the parent, assume it's backers if it's just a number
                                if text.isdigit() and len(text) < 8:  # Not too long to be something else
                                    campaign_details['backers_count'] = text
                                    logger.info(f"Extracted likely backers count: {text}")
                                    backers_found = True
                                    break
                except Exception as e:
                    logger.error(f"Error with direct extraction of live project backers: {str(e)}")
                    
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
                            backers_found = True
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
        
        # First extract campaign details from the main page
        campaign_details = extract_campaign_details(browser)
        logger.info("Campaign details extraction complete")
        
        # Make sure we're still on the main page after extracting campaign details
        if browser.current_url != main_page_url:
            logger.info(f"Navigating back to main page: {main_page_url}")
            browser.get(main_page_url)
            WebDriverWait(browser, 10).until(
                lambda x: x.execute_script("return document.readyState") == "complete"
            )
            random_sleep(2, 3)
        
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
            # Return both campaign details and empty updates
            return {"campaign_details": campaign_details, "updates": []}

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
                logger.error("Still failed to navigate to updates page. Returning campaign details only.")
                return {"campaign_details": campaign_details, "updates": []}
        
        # Check for project launch date on the updates page if it wasn't found on the main page
        if 'funding_start_date' not in campaign_details:
            try:
                logger.info("Looking for project launch date on updates page...")
                
                # Selectors for project launch information based on the image
                launch_selectors = [
                    "div.flex.justify-center.items-center.text-center.white.bp-ksr-green",
                    "div.flex.flex-column.items-center.justify-center.text-center",
                    "div[class*='flex'][class*='justify-center']",
                    "div.flex span.type-16",
                    "div.flex p.type-16",
                    # Target exactly what was shown in the image
                    "div.flex-column.items-center.justify-center div.type-16",
                    "div.type-16.type-24-sm.bold.pb5",
                    # Most specific selector for "Project launches"
                    "div.type-16.type-24-sm.bold.pb5:contains('Project launches')"
                ]
                
                for selector in launch_selectors:
                    try:
                        # Try CSS selector first, fallback to XPath for contains
                        if ":contains" in selector:
                            # Extract base selector and text to find
                            base_selector, text_to_find = selector.split(":contains")
                            text_to_find = text_to_find.strip("(')").lower()
                            
                            # Use XPath contains instead
                            elements = browser.find_elements(By.XPATH, 
                                f"//*[contains(@class, '{base_selector.replace('.', ' ').strip()}') and contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text_to_find}')]")
                        else:
                            elements = browser.find_elements(By.CSS_SELECTOR, selector)
                        
                        for element in elements:
                            if "launch" in element.text.lower():
                                launch_text = element.text.strip()
                                logger.info(f"Found launch text: {launch_text}")
                                
                                # Look for date text in this element or nearby elements
                                date_element = None
                                # First check if the date is in the same element
                                if any(month.lower() in launch_text.lower() for month in [
                                    "january", "february", "march", "april", "may", "june", 
                                    "july", "august", "september", "october", "november", "december"]):
                                    date_element = element
                                else:
                                    # Look for date in nearby elements
                                    try:
                                        # 1. Look specifically for uppercase date text as shown in the image
                                        # Look for the specific format in the image: "FEBRUARY 18, 2025"
                                        uppercase_date_selectors = [
                                            "div.type-11.type-14-sm.text-uppercase",  # Exactly matching the image
                                            "div.text-uppercase",
                                            "span.text-uppercase",
                                            "*.text-uppercase"  # Any element with text-uppercase class
                                        ]
                                        
                                        for uppercase_selector in uppercase_date_selectors:
                                            try:
                                                uppercase_elements = element.find_elements(By.CSS_SELECTOR, uppercase_selector)
                                                if not uppercase_elements:  # If not found in this element, try browser-wide
                                                    uppercase_elements = browser.find_elements(By.CSS_SELECTOR, uppercase_selector)
                                                
                                                for up_elem in uppercase_elements:
                                                    up_text = up_elem.text.strip()
                                                    if any(month.lower() in up_text.lower() for month in [
                                                        "january", "february", "march", "april", "may", "june", 
                                                        "july", "august", "september", "october", "november", "december"]):
                                                        logger.info(f"Found uppercase date text: {up_text}")
                                                        date_element = up_elem
                                                        break
                                                
                                                if date_element:
                                                    break
                                            except Exception as e:
                                                logger.error(f"Error with uppercase selector {uppercase_selector}: {str(e)}")
                                                continue
                                        
                                        # 2. If not found with uppercase selectors, check immediate siblings
                                        if not date_element:
                                            siblings = element.find_elements(By.XPATH, "./following-sibling::*")
                                            siblings.extend(element.find_elements(By.XPATH, "./preceding-sibling::*"))
                                            
                                            # 3. Check parent's siblings
                                            parent = element.find_element(By.XPATH, "..")
                                            siblings.extend(parent.find_elements(By.XPATH, "./following-sibling::*"))
                                            siblings.extend(parent.find_elements(By.XPATH, "./preceding-sibling::*"))
                                            
                                            # 4. Check children
                                            siblings.extend(element.find_elements(By.XPATH, "./*"))
                                            
                                            for sibling in siblings:
                                                sibling_text = sibling.text.strip().lower()
                                                if any(month.lower() in sibling_text for month in [
                                                    "january", "february", "march", "april", "may", "june", 
                                                    "july", "august", "september", "october", "november", "december"]):
                                                    date_element = sibling
                                                    break
                                    except Exception as e:
                                        logger.error(f"Error finding date element: {str(e)}")
                                
                                if date_element:
                                    date_text = date_element.text.strip()
                                    logger.info(f"Found date text: {date_text}")
                                    
                                    # Extract and parse the date
                                    import re
                                    from datetime import datetime
                                    
                                    # Match patterns like "FEBRUARY 18, 2025"
                                    date_patterns = [
                                        r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}',
                                        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{4}'
                                    ]
                                    
                                    date_match = None
                                    for pattern in date_patterns:
                                        matches = re.finditer(pattern, date_text.lower(), re.IGNORECASE)
                                        for match in matches:
                                            date_match = match.group(0)
                                            break
                                        if date_match:
                                            break
                                    
                                    if not date_match and 'launch' in date_text.lower():
                                        # The date might be in the same text as "Project launches"
                                        for pattern in date_patterns:
                                            matches = re.finditer(pattern, date_text.lower(), re.IGNORECASE)
                                            for match in matches:
                                                date_match = match.group(0)
                                                break
                                            if date_match:
                                                break
                                    
                                    if date_match:
                                        # Try to parse the date
                                        try:
                                            # Try various date formats
                                            date_formats = [
                                                "%B %d, %Y",  # February 18, 2025
                                                "%B %d %Y",   # February 18 2025
                                                "%b %d, %Y",  # Feb 18, 2025
                                                "%b %d %Y"    # Feb 18 2025
                                            ]
                                            
                                            parsed_date = None
                                            for date_format in date_formats:
                                                try:
                                                    # Handle case insensitivity
                                                    parsed_date = datetime.strptime(date_match.strip(), date_format)
                                                    break
                                                except:
                                                    continue
                                            
                                            if parsed_date:
                                                # Format the date in the expected ISO format
                                                funding_start_date = parsed_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                                                campaign_details['funding_start_date'] = funding_start_date
                                                logger.info(f"Extracted funding start date from updates page: {funding_start_date}")
                                                
                                                # Also store the raw text for reference
                                                campaign_details['project_launch_text'] = f"{launch_text} {date_text}".replace("\n", " ").strip()
                                                break
                                        except Exception as e:
                                            logger.error(f"Error parsing date: {str(e)}")
                                    
                                # Break out of the selector loop if we found the date
                                if 'funding_start_date' in campaign_details:
                                    break
                    except Exception as e:
                        logger.error(f"Error with launch selector {selector}: {str(e)}")
                        continue
                
                # If we still haven't found the funding start date, try a more general approach
                if 'funding_start_date' not in campaign_details:
                    try:
                        # Look for any elements containing both "project" and "launches"
                        project_elements = browser.find_elements(By.XPATH, "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'project') and contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'launch')]")
                        
                        for elem in project_elements:
                            elem_text = elem.text.strip()
                            if "launch" in elem_text.lower() and any(month.lower() in elem_text.lower() for month in [
                                "january", "february", "march", "april", "may", "june", 
                                "july", "august", "september", "october", "november", "december"]):
                                
                                logger.info(f"Found project launch text: {elem_text}")
                                campaign_details['project_launch_text'] = elem_text.replace("\n", " ").strip()
                                
                                # Extract date using regex
                                import re
                                from datetime import datetime
                                
                                date_pattern = r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{4}'
                                date_match = re.search(date_pattern, elem_text.lower(), re.IGNORECASE)
                                
                                if date_match:
                                    date_str = date_match.group(0)
                                    logger.info(f"Extracted date string: {date_str}")
                                    
                                    # Try to parse the date
                                    date_formats = [
                                        "%B %d, %Y",  # February 18, 2025
                                        "%B %d %Y",   # February 18 2025
                                        "%b %d, %Y",  # Feb 18, 2025
                                        "%b %d %Y"    # Feb 18 2025
                                    ]
                                    
                                    for date_format in date_formats:
                                        try:
                                            parsed_date = datetime.strptime(date_str, date_format)
                                            funding_start_date = parsed_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                                            campaign_details['funding_start_date'] = funding_start_date
                                            logger.info(f"Extracted funding start date from text: {funding_start_date}")
                                            break
                                        except:
                                            continue
                                
                                # Break if we found and parsed the date
                                if 'funding_start_date' in campaign_details:
                                    break
                    except Exception as e:
                        logger.error(f"Error with general project launch search: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error extracting project launch date from updates page: {str(e)}")
        
        # If we found a funding_start_date but not funding_period_text, create one
        if 'funding_start_date' in campaign_details and 'funding_end_date' in campaign_details and 'funding_period_text' not in campaign_details:
            try:
                from datetime import datetime
                
                # Convert ISO format to readable date
                start_date = datetime.fromisoformat(campaign_details['funding_start_date'].replace('Z', '+00:00'))
                end_date = datetime.fromisoformat(campaign_details['funding_end_date'].replace('Z', '+00:00'))
                
                # Format dates as "Feb 4 2025 - Mar 6 2025"
                readable_start = start_date.strftime("%b %d %Y")
                readable_end = end_date.strftime("%b %d %Y")
                
                # Calculate duration in days
                duration_days = (end_date - start_date).days
                
                # Create funding period text
                funding_period_text = f"Funding period\n{readable_start} - {readable_end} ({duration_days} days)"
                campaign_details['funding_period_text'] = funding_period_text
                campaign_details['funding_duration_days'] = duration_days
                
                logger.info(f"Created funding period text: {funding_period_text}")
            except Exception as e:
                logger.error(f"Error creating funding period text: {str(e)}")
        
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
                        return {"campaign_details": campaign_details, "updates": []}
            except Exception as e:
                logger.error(f"Error counting updates on the page: {str(e)}")
                # If we can't determine the count, assume there are no updates
                return {"campaign_details": campaign_details, "updates": []}

        max_attempts = updates_count + 2 if updates_count > 0 else 5
        attempt = 0
        consecutive_failures = 0
        previous_update_count = 0
        
        while len(updates) < updates_count and attempt < max_attempts and consecutive_failures < 2:
            attempt += 1
            previous_update_count = len(updates)
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
                    # Try one more time with a generic approach
                    try:
                        # Try to find any container that might have updates
                        containers = browser.find_elements(By.CSS_SELECTOR, "div.grid-container, div.post-index, div.posts-container")
                        if containers:
                            updates_container = containers[0]
                            logger.info("Found a potential updates container as fallback")
                        else:
                            # Last resort: use the body element
                            updates_container = browser.find_element(By.TAG_NAME, "body")
                            logger.info("Using body element as fallback container")
                    except Exception as e:
                        logger.error(f"Error in fallback container search: {str(e)}")
                        break
                
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
                
                # At the end of the loop, check if we found any new updates
                if len(updates) == previous_update_count:
                    consecutive_failures += 1
                    logger.info(f"No new updates found in this attempt. Consecutive failures: {consecutive_failures}")
                else:
                    consecutive_failures = 0  # Reset counter if we found new updates
                    logger.info(f"Found {len(updates) - previous_update_count} new updates in this attempt")
                
                if consecutive_failures >= 1:
                    logger.info("Stopping update extraction after 2 consecutive unsuccessful attempts")
                    break
                
                if len(updates) < updates_count:
                    logger.info("Scrolling to load more updates...")
                    human_like_scroll(browser)
                    random_sleep(2, 3)
                
            except Exception as e:
                logger.error(f"Error in update processing loop: {str(e)}")
                consecutive_failures += 1
                continue

    except Exception as e:
        logger.error(f"Error in extract_updates_content: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    logger.info(f"Finished processing updates. Found {len(updates)} updates.")
    
    # Standardize the campaign details format to match the expected JSON structure
    standardized_campaign_details = {}
    
    # Follow the order from the example: funding_period_text, funding_start_date, funding_end_date, 
    # funding_duration_days, pledged_amount, funding_goal, backers_count
    standard_keys = [
        'funding_period_text', 
        'funding_start_date', 
        'funding_end_date', 
        'funding_duration_days',
        'pledged_amount', 
        'funding_goal', 
        'backers_count'
    ]
    
    # Add required keys in the standard order
    for key in standard_keys:
        if key in campaign_details:
            standardized_campaign_details[key] = campaign_details[key]
    
    # Add any remaining keys not in the standard list
    for key, value in campaign_details.items():
        if key not in standard_keys:
            standardized_campaign_details[key] = value
    
    # Return both standardized campaign details and updates
    return {"campaign_details": standardized_campaign_details, "updates": updates} 