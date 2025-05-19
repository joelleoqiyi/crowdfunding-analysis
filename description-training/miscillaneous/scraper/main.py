import yaml
from webscraper import scrape_kickstarter_details
from data_processing import process_scraped_data
from file_handler import save_to_csv

# Load config
with open("C:/Users/alyss/crowd-funding-analysis/scraper/config.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

print("Config file loaded successfully!")

batch_size = config["batch_size"]

for category, details in config["categories"].items():
    url_base = details["url"]
    last_scraped_page = details["last_scraped_page"]
    total_pages = details["total_pages"]
    output_file = config[f"output_file_{category}"]

    # Determine the next batch range
    start_page = last_scraped_page + 1
    end_page = min(start_page + batch_size - 1, total_pages)

    if start_page > total_pages:
        print(f"[INFO] {category.capitalize()} campaigns already fully scraped.")
        continue

    print(f"[INFO] Scraping {category} campaigns from page {start_page} to {end_page}")

    all_data = []
    for page in range(start_page, end_page + 1):
        url = f"{url_base}{page}"
        page_data = scrape_kickstarter_details(url)
        if page_data:
            all_data.extend(page_data)

    if all_data:
        df = process_scraped_data(all_data)
        save_to_csv(df, output_file)

        # Update last scraped page in config
        config["categories"][category]["last_scraped_page"] = end_page
        with open("config.yaml", "w") as config_file:
            yaml.dump(config, config_file)

        print(f"[SUCCESS] Data saved. Updated last_scraped_page for {category} to {end_page}.")
    else:
        print(f"[WARNING] No data extracted for {category}. CSV not updated.")
