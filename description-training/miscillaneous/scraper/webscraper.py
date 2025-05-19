from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import json
from datetime import datetime 


def scrape_kickstarter_details(url):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")  # Disable GPU acceleration
    options.add_argument("--no-sandbox")  # Disable sandboxing
    options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".js-react-proj-card.grid-col-12.grid-col-6-sm.grid-col-4-lg")))

        # Allow JavaScript to load content
        time.sleep(5)

        # Select project containers
        project_containers = driver.find_elements(By.CSS_SELECTOR, ".js-react-proj-card.grid-col-12.grid-col-6-sm.grid-col-4-lg")

        if not project_containers:
            print(f"[ERROR] No project containers found on {url}. Skipping page.")
            return []

        extracted_data = []  # Store multiple projects for this page

        for container in project_containers:
            data_project = container.get_attribute("data-project")  # Get JSON string

            if not data_project:
                continue  # Skip this project

            try:
                project_info = json.loads(data_project)  # Convert JSON string to dictionary
            except json.JSONDecodeError:
                print("[ERROR] Failed to parse JSON data. Skipping this project.")
                continue  # Skip invalid JSON entry

            # Extract only the required keys
            project_data = {
                "name": project_info.get("name", "N/A"),
                "subtitle": project_info.get("blurb", "N/A"),
                "backers": project_info.get("backers_count", "N/A"),
                "goal": project_info.get("goal", "N/A"),
                "pledged": project_info.get("pledged", "N/A"),
                "usd_pledged": project_info.get("usd_pledged", "N/A"),
                "created_at": project_info.get("created_at", "N/A"),
                "launched_at": project_info.get("launched_at", "N/A"),
                "deadline": project_info.get("deadline", "N/A"),
                "state_changed_at": project_info.get("state_changed_at", "N/A"),
                "category": project_info.get("category", {}).get("name", "N/A"),
                "location": project_info.get("location", {}).get("name", "N/A"),
                "staff_pick": project_info.get("staff_pick", "N/A"),
                "is_in_post_campaign_pledging_phase": project_info.get("is_in_post_campaign_pledging_phase", "N/A"),
                "state": project_info.get("state", "N/A")
            }

            extracted_data.append(project_data)

        print(f"Scraping complete for {url}. {len(extracted_data)} projects extracted.")

    except Exception as e:
        print(f"[ERROR] Something went wrong on {url}: {e}")
        extracted_data = []

    finally:
        driver.quit()

    return extracted_data



