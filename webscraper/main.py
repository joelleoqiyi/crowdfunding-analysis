import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger_config import setup_logging
from scrapers.project_scraper import scrape_project, scrape_technology_projects

def read_links(file_path):
    """Read links and their update counts from file"""
    with open(file_path, 'r') as f:
        return [(line.strip().split('\t')[0], int(line.strip().split('\t')[1])) 
                for line in f if line.strip()]

def move_link_to_previous(link, update_count, links_file, previous_file):
    """Move a successfully scraped link from current to previous links file"""
    # Read all current links
    with open(links_file, 'r') as f:
        links = f.readlines()
    
    # Write back all links except the processed one
    with open(links_file, 'w') as f:
        for l in links:
            if l.strip().split('\t')[0] != link:
                f.write(l)
    
    # Append to previously scraped links
    with open(previous_file, 'a') as f:
        f.write(f"{link}\t{update_count}\n")

def main():
    logger = setup_logging()
    
    script_dir = os.path.dirname(__file__)
    links_file = os.path.join(script_dir, 'unscraped_links.txt')
    previous_file = os.path.join(script_dir, 'previously_scraped_links.txt')
    
    try:
        links = read_links(links_file)
        logger.info(f"Found {len(links)} links to process")
        
        # Sort links by update count in descending order
        links.sort(key=lambda x: x[1], reverse=True)
        
        for i, (url, update_count) in enumerate(links, 1):
            logger.info(f"\nProcessing link {i}/{len(links)}")
            logger.info(f"URL: {url} (Updates: {update_count})")
            
            try:
                result = scrape_project(url)
                logger.info(f"Successfully scraped project")
                logger.info(f"Updates count: {result['updates']['count']}")
                
                # Move successfully scraped link to previously_scraped_links.txt
                move_link_to_previous(url, update_count, links_file, previous_file)
                logger.info(f"Moved {url} to previously scraped links")
                
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