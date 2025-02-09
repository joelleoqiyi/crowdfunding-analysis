from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd

def scrape_kickstarter_details(url):
    options = Options()
    options.add_argument("--headless")  # Run without opening a browser window
    options.add_argument("--disable-blink-features=AutomationControlled")  # Prevent Selenium detection
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    wait = WebDriverWait(driver, 10)  # Wait up to 10 seconds for elements to appear

    data = {}

    # Ensure the page loads properly
    time.sleep(5)  # Allow time for JavaScript to render content

    # Project Name (using JavaScript)
    try:
        data['project_name'] = driver.execute_script(
            "return document.querySelector('h1.project-name') ? document.querySelector('h1.project-name').innerText.trim() : 'N/A';"
        )
    except:
        data['project_name'] = "N/A"

    # Subtitle (using JavaScript)
    try:
        data['subtitle'] = driver.execute_script(
            "return document.querySelector('p.project-description') ? document.querySelector('p.project-description').innerText.trim() : 'N/A';"
        )
    except:
        data['subtitle'] = "N/A"

    # Story (description with images)
    try:
        story_elements = driver.find_elements(By.CLASS_NAME, "rte__content")
        story_text = " ".join([elem.text.strip() for elem in story_elements])
        story_images = [img.get_attribute("src") for img in driver.find_elements(By.TAG_NAME, "img")]
        data['story'] = {"text": story_text, "images": story_images}
    except:
        data['story'] = "N/A"

    # Amount Backed
    try:
        amount_backed_element = driver.find_element(By.CLASS_NAME, "ksr-green-500")
        data['amount_backed'] = amount_backed_element.text.strip()
    except:
        data['amount_backed'] = "N/A"

    # Total Goal Amount
    try:
        total_goal_element = driver.find_element(By.CLASS_NAME, "inline-block-sm")
        data['total_goal'] = total_goal_element.text.split("pledged of ")[1].split(" goal")[0].strip()
    except:
        data['total_goal'] = "N/A"

    # Number of Backers
    try:
        backers_element = driver.find_element(By.CSS_SELECTOR, "div.mb4-lg span.bold")
        data['backers'] = backers_element.text.strip()
    except:
        data['backers'] = "N/A"

    # Days to Go
    try:
        days_to_go_element = driver.find_element(By.CSS_SELECTOR, "div.ml5 span.bold")
        data['days_to_go'] = days_to_go_element.text.strip()
    except:
        data['days_to_go'] = "N/A"

    # Number of Other Campaigns Created
    try:
        campaigns_created_element = driver.execute_script(
            "return document.querySelector('button div.text-left.mb3') ? document.querySelector('button div.text-left.mb3').innerText.split(' created')[0].trim() : 'N/A';"
        )
        data['campaigns_created'] = campaigns_created_element
    except:
        data['campaigns_created'] = "N/A"

    print("Scraping complete. Browser running in headless mode.")
    driver.quit()
    return data

# Example Usage
url = "https://www.kickstarter.com/projects/connernyberg/newgrounds-documentary/"
kickstarter_data = scrape_kickstarter_details(url)
print(kickstarter_data)

# Convert to DataFrame and save as CSV
df = pd.DataFrame([kickstarter_data])
df.to_csv("kickstarter_details.csv", index=False)
