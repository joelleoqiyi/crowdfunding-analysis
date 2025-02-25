import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger_config import setup_logging
from scrapers.project_scraper import scrape_project, scrape_technology_projects

def read_links(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    logger = setup_logging()
    
    links_file = os.path.join(os.path.dirname(__file__), 'scraped_links.txt')
    
    try:
        links = read_links(links_file)
        logger.info(f"Found {len(links)} links to process")
        
        for i, url in enumerate(links, 1):
            logger.info(f"\nProcessing link {i}/{len(links)}")
            logger.info(f"URL: {url}")
            
            try:
                result = scrape_project(url)
                logger.info(f"Successfully scraped project")
                logger.info(f"Updates count: {result['updates']['count']}")
                
                if result['updates']['content']:
                    logger.info("First update example:")
                    logger.info(result['updates']['content'][0])
                    
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                continue
            
    except Exception as e:
        logger.error(f"Error reading links file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 