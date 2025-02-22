import os
import sys

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger_config import setup_logging
from scrapers.project_scraper import scrape_project, scrape_technology_projects

def main():
    logger = setup_logging()
    
    # Test with a single project
    test_url = "https://www.kickstarter.com/projects/16159264/xlaser-the-ultimate-4-in-1-laser-welding-revolution"
    result = scrape_project(test_url)
    
    logger.info("\nFinal results:")
    logger.info(f"URL: {result['url']}")
    logger.info(f"Updates count: {result['updates']['count']}")
    
    if result['updates']['content']:
        logger.info("\nFirst update example:")
        logger.info(result['updates']['content'][0])
    
   

if __name__ == "__main__":
    main() 