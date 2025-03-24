import os
import sys
import json

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
                
                # Log campaign details
                if 'campaign_details' in result and result['campaign_details']:
                    campaign_details = result['campaign_details']
                    logger.info(f"Campaign details: Pledged: {campaign_details.get('pledged_amount', 'N/A')}, " +
                               f"Goal: {campaign_details.get('funding_goal', 'N/A')}, " +
                               f"Backers: {campaign_details.get('backers_count', 'N/A')}")
                
                # Log updates
                updates_count = result['updates']['count'] if 'updates' in result else 0
                logger.info(f"Updates count: {updates_count}")
                
                # Move successfully scraped link to previously_scraped_links.txt
                move_link_to_previous(url, update_count, links_file, previous_file)
                logger.info(f"Moved {url} to previously scraped links")
                
                # Show example of first update if available
                if 'updates' in result and result['updates']['content'] and len(result['updates']['content']) > 0:
                    logger.info("First update example:")
                    first_update = result['updates']['content'][0]
                    logger.info(f"Title: {first_update.get('title', 'N/A')}")
                    logger.info(f"Date: {first_update.get('date', 'N/A')}")
                    logger.info(f"Content preview: {first_update.get('content', 'N/A')[:100]}...")
                    
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                continue
            
    except Exception as e:
        logger.error(f"Error reading links file: {str(e)}")
        sys.exit(1)

def test_single_project():
    """Test scraping a single project"""
    logger = setup_logging()
    
    # Replace with the URL you want to test
    test_url = "https://www.kickstarter.com/projects/circular-ring/circular-ring-2-worlds-most-advanced-health-tracking-ring"
    
    logger.info(f"Testing scraping of a single project: {test_url}")
    
    try:
        result = scrape_project(test_url)
        
        # Log data length as a simple check
        data_length = len(json.dumps(result))
        logger.info(f"Successfully scraped project. Data length: {data_length} characters")
        
        # Log campaign details
        if 'campaign_details' in result and result['campaign_details']:
            campaign_details = result['campaign_details']
            logger.info(f"Campaign details: Pledged: {campaign_details.get('pledged_amount', 'N/A')}, " +
                       f"Goal: {campaign_details.get('funding_goal', 'N/A')}, " +
                       f"Backers: {campaign_details.get('backers_count', 'N/A')}")
        
        # Log metrics
        logger.info(f"Project metrics: {result.get('metrics', {})}")
        
        # Log updates count
        updates_count = result['updates']['count'] if 'updates' in result else 0
        logger.info(f"Updates count: {updates_count}")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")

if __name__ == "__main__":
    import sys
    
    # Check if a command line argument is provided
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run the test function if "test" argument is provided
        test_single_project()
    else:
        # Run the main function by default
        main() 