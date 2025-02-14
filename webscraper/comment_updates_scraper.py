from selenium.webdriver.common.by import By
import time

class CommentsUpdatesScraper:
    def __init__(self, driver):
        self.__driver = driver

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

    def get_updates(self):
        browser = self.__driver
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
        
    def extract_updates_content(self):
        browser = self.__driver
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
    

    def extract_comments_content(self):
        browser = self.__driver
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