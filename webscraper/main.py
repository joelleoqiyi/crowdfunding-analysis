import os
import sys
import json
import argparse

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

def scrape_from_links_file():
    """Scrape projects from a links file"""
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

def test_single_project(url=None):
    """Test scraping a single project"""
    logger = setup_logging()
    
    # Replace with the URL you want to test
    test_url = "https://www.kickstarter.com/projects/-vision-prime-1/sirui-vision-prime-1-t14-cine-lens-series"
    
    logger.info(f"Testing scraping of a single project: {test_url}")
    
    try:
        result = scrape_project(test_url)
        
        # Log data length as a simple check
        data_length = len(json.dumps(result))
        logger.info(f"Successfully scraped project. Data length: {data_length} characters")
        
        # Log campaign details
        if 'campaign_details' in result and result['campaign_details']:
            campaign_details = result['campaign_details']
            logger.info(f"Campaign details:")
            logger.info(f"  Pledged: {campaign_details.get('pledged_amount', 'N/A')}")
            logger.info(f"  Goal: {campaign_details.get('funding_goal', 'N/A')}")
            logger.info(f"  Backers: {campaign_details.get('backers_count', 'N/A')}")
            logger.info(f"  Funding period: {campaign_details.get('funding_period', 'N/A')}")
            logger.info(f"  Days remaining: {campaign_details.get('days_remaining', 'N/A')}")
            logger.info(f"  Duration (days): {campaign_details.get('funding_duration_days', 'N/A')}")
        
        # Log updates count
        updates_count = result['updates']['count'] if 'updates' in result else 0
        logger.info(f"Updates count: {updates_count}")
        
        # Show example of first update if available
        if 'updates' in result and result['updates']['content'] and len(result['updates']['content']) > 0:
            logger.info("First update example:")
            first_update = result['updates']['content'][0]
            logger.info(f"Title: {first_update.get('title', 'N/A')}")
            logger.info(f"Date: {first_update.get('date', 'N/A')}")
            logger.info(f"Content preview: {first_update.get('content', 'N/A')[:100]}...")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")

def scrape_technology_category(start_page=1, max_pages=5):
    """Scrape projects from the technology category"""
    logger = setup_logging()
    
    logger.info(f"Scraping technology projects from page {start_page} to {start_page + max_pages - 1}")
    try:
        projects = scrape_technology_projects(start_page=start_page, max_pages=max_pages)
        logger.info(f"Successfully scraped {len(projects)} technology projects")
    except Exception as e:
        logger.error(f"Error in technology category scraper: {str(e)}")
        sys.exit(1)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Kickstarter Project Scraper')
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Scraper mode')
    
    # Single project testing
    test_parser = subparsers.add_parser('test', help='Test scrape a single project')
    test_parser.add_argument('--url', help='URL of the project to test')
    
    # Category scraping
    category_parser = subparsers.add_parser('category', help='Scrape a category of projects')
    category_parser.add_argument('--start-page', type=int, default=1, help='Page to start scraping from')
    category_parser.add_argument('--max-pages', type=int, default=5, help='Number of pages to scrape')
    
    # Links file scraping
    links_parser = subparsers.add_parser('links', help='Scrape projects from links file')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    if args.mode == 'test':
        test_single_project(args.url)
    elif args.mode == 'category':
        scrape_technology_category(args.start_page, args.max_pages)
    elif args.mode == 'links':
        scrape_from_links_file()
    else:
        # Default to category scraping if no mode specified
        scrape_technology_category()

if __name__ == "__main__":
    main() 