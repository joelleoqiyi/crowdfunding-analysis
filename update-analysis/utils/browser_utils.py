import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from fake_useragent import UserAgent
import random
import time
import logging

logger = logging.getLogger(__name__)

def random_sleep(min_time=1, max_time=3):
    """Sleep for a random amount of time between min_time and max_time"""
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
    
    browser = uc.Chrome(options=options, version_main=133)
    
    # Additional stealth measures
    browser.execute_cdp_cmd('Network.setUserAgentOverride', {
        "userAgent": ua.random
    })
    
    # Modify navigator.webdriver flag
    browser.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return browser 