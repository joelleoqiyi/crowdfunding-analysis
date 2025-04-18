from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

def scrape_kickstarter(url):
    # Configure Selenium for headless browsing
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        driver.get(url)
        time.sleep(3)  # Wait for dynamic content to load (adjust as needed)
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Example: extract the campaign description
        description = soup.find('div', class_='full-description-class')  # Use the actual class/id
        if description:
            description_text = description.get_text(strip=True)
        else:
            description_text = ""
        
        # Similarly, extract updates and comments as needed
        # ...
        
        return {
            "description": description_text,
            # "updates": updates_text,
            # "comments": comments_text
        }
    finally:
        driver.quit()

# Example usage in a Flask route:
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/scrape', methods=['POST'])
def scrape_route():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    scraped_data = scrape_kickstarter(url)
    # Pass scraped_data to your model for prediction
    # prediction = model.predict(scraped_data)
    return jsonify({
        "scraped_data": scraped_data,
        # "prediction": prediction
    })

if __name__ == '__main__':
    app.run(debug=True)