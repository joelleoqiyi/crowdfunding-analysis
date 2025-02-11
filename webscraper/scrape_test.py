from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def get_updates(browser):
    try:
        xpath_patterns = [
            "//div[@class='row']//a[@data-content='updates']//span",
            "//a[@data-content='updates']//span",
            "//a[contains(@class, 'js-load-project-updates')]//span",
            "//a[contains(@class, 'js-load-project-updates')]",
            "//a[contains(@href, '/updates')]//span"
        ]
        
        for xpath in xpath_patterns:
            try:
                element = browser.find_element(By.XPATH, xpath)
                print(f"Found updates element with XPath: {xpath}")
                print(f"Element text: {element.text}")
                return int(''.join(filter(str.isdigit, element.text)))
            except Exception as e:
                print(f"Failed with XPath {xpath}: {str(e)}")
                continue
                
        print("Could not find updates count with any known XPath pattern")
        return 0
    except Exception as e:
        print(f"Error in get_updates: {str(e)}")
        return 0

def get_comments(browser):
    try:
        xpath_patterns = [
            "//div[@class='row']//a[@data-content='comments']//span//data",
            "//a[@data-content='comments']//span//data",
            "//a[contains(@class, 'js-load-project-comments')]//span",
            "//a[contains(@href, '/comments')]//span",
            "//a[contains(@data-content, 'comments')]//span"
        ]
        
        for xpath in xpath_patterns:
            try:
                element = browser.find_element(By.XPATH, xpath)
                print(f"Found comments element with XPath: {xpath}")
                print(f"Element text: {element.text}")
                return int(''.join(filter(str.isdigit, element.text)))
            except Exception as e:
                print(f"Failed with XPath {xpath}: {str(e)}")
                continue
                
        print("Could not find comments count with any known XPath pattern")
        return 0
    except Exception as e:
        print(f"Error in get_comments: {str(e)}")
        return 0

def extract_updates_content(browser):
    updates = []
    try:
        updates_tab = browser.find_element(By.CSS_SELECTOR, "a[data-content='updates']")
        browser.execute_script("arguments[0].click();", updates_tab)
        print("Clicked updates tab")
        time.sleep(5)  

        last_height = browser.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_attempts = 10  

        while scroll_attempts < max_attempts:
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = browser.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            scroll_attempts += 1

        # Click all "Read more" buttons to expand and get details for each update
        read_more_buttons = browser.find_elements(By.CSS_SELECTOR, "button[aria-label='Read more']")
        for button in read_more_buttons:
            try:
                browser.execute_script("arguments[0].click();", button)
                time.sleep(1)
            except:
                continue  

        update_posts = browser.find_elements(By.CSS_SELECTOR, "article.timeline-post")
        print(f"Found {len(update_posts)} update posts")

        for post in update_posts:
            try:
                title = post.find_element(By.CSS_SELECTOR, "h2").text.strip()
                date = post.find_element(By.CSS_SELECTOR, "time").get_attribute('datetime')
                content = post.find_element(By.CSS_SELECTOR, "div.rte__content").text.strip()

                updates.append({
                    'title': title,
                    'date': date,
                    'content': content
                })
                print(f"Extracted update: {title[:50]}...")
            except Exception as e:
                print(f"Error extracting update post: {str(e)}")

    except Exception as e:
        print(f"Error extracting updates: {str(e)}")

    return updates

def extract_comments_content(browser):
    comments = []
    try:
        comments_tab = browser.find_element(By.CSS_SELECTOR, "a[data-content='comments']")
        browser.execute_script("arguments[0].click();", comments_tab)
        print("Clicked comments tab")
        time.sleep(5)  

        last_height = browser.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        max_attempts = 10  

        while scroll_attempts < max_attempts:
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = browser.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            scroll_attempts += 1

        comment_elements = browser.find_elements(By.CSS_SELECTOR, "li.comment-item")
        print(f"Found {len(comment_elements)} comment elements")

        for comment in comment_elements:
            try:
                author = comment.find_element(By.CSS_SELECTOR, "a.author").text.strip()
                date = comment.find_element(By.CSS_SELECTOR, "time").get_attribute('datetime')
                content = comment.find_element(By.CSS_SELECTOR, "p.comment-content").text.strip()

                comments.append({
                    'author': author,
                    'date': date,
                    'content': content
                })
                print(f"Extracted comment from: {author}")
            except Exception as e:
                print(f"Error extracting comment: {str(e)}")

    except Exception as e:
        print(f"Error extracting comments: {str(e)}")

    return comments


def scrape_project(url):
    options = webdriver.ChromeOptions()
    #options.add_argument("--headless")  # Run without opening a browser window
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    wait = WebDriverWait(driver, 20)

    try:
        print(f"\nProcessing project: {url}")
        
        driver.get(url)
        print("Page loaded")
        
        time.sleep(5)
        driver.execute_script("window.scrollTo(0, 500)")
        time.sleep(2)
        
        print(f"Page title: {driver.title}")
        
        updates_count = get_updates(driver)
        comments_count = get_comments(driver)
        
        print(f"Final updates count: {updates_count}")
        print(f"Final comments count: {comments_count}")
        
        # Extract updates and comments content
        print("\nExtracting updates content...")
        updates_content = extract_updates_content(driver)
        print(f"Found {len(updates_content)} updates")
        
        print("\nExtracting comments content...")
        comments_content = extract_comments_content(driver)
        print(f"Found {len(comments_content)} comments")
        
        return {
            'url': url,
            'updates': {
                'count': updates_count,
                'content': updates_content
            },
            'comments': {
                'count': comments_count,
                'content': comments_content
            }
        }
        
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return {
            'url': url,
            'updates': {'count': 0, 'content': []},
            'comments': {'count': 0, 'content': []},
            'error': str(e)
        }
    finally:
        driver.quit()

if __name__ == "__main__":
    #this is link to a project I tested with, can change 
    test_url = "https://www.kickstarter.com/projects/halliday-ai-glasses/halliday-proactive-ai-glasses-with-invisible-display"
    result = scrape_project(test_url)
    
    print("\nFinal results:")
    print(f"URL: {result['url']}")
    print(f"Updates count: {result['updates']['count']}")
    print(f"Number of updates extracted: {len(result['updates']['content'])}")
    print(f"Comments count: {result['comments']['count']}")
    print(f"Number of comments extracted: {len(result['comments']['content'])}")
    
    if result['updates']['content']:
        print("\nFirst update example:")
        print(result['updates']['content'][0])
    
    if result['comments']['content']:
        print("\nFirst comment example:")
        print(result['comments']['content'][0]) 