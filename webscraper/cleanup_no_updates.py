import os
import json
import logging
from utils.logger_config import setup_logging

def get_base_filename(filename):
    # Remove the timestamp portion (assumes format: name_YYYYMMDD_HHMMSS.json)
    return '_'.join(filename.split('_')[:-2]) if '_' in filename else filename

def cleanup_no_update_files():
    logger = setup_logging()
    
    # Get the correct scraped data directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'scrapers', 'scraped_data_failed')
    
    if not os.path.exists(data_dir):
        logger.error(f"Scraped data directory not found at {data_dir}")
        return
        
    # Count statistics
    total_files = 0
    deleted_no_updates = 0
    deleted_duplicates = 0
    
    # Track unique projects by base filename
    seen_projects = {}  
    
    # First pass: collect all files and group by base name
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue
            
        total_files += 1
        base_name = get_base_filename(filename)
        
        if base_name in seen_projects:
            seen_projects[base_name].append(filename)
        else:
            seen_projects[base_name] = [filename]
    
    # Second pass: process each group of files
    for base_name, filenames in seen_projects.items():
        if len(filenames) > 1:
            # Sort by timestamp (newest first)
            filenames.sort(reverse=True)
            
            # Keep the newest file, delete the rest
            for filename in filenames[1:]:
                file_path = os.path.join(data_dir, filename)
                try:
                    logger.info(f"Deleting duplicate {filename} - Keeping {filenames[0]}")
                    os.remove(file_path)
                    deleted_duplicates += 1
                except Exception as e:
                    logger.error(f"Error deleting {filename}: {str(e)}")
        
        # # Check the remaining file for updates - COMMENTED OUT
        # newest_file = filenames[0]
        # file_path = os.path.join(data_dir, newest_file)
        # try:
        #     with open(file_path, 'r', encoding='utf-8') as f:
        #         data = json.load(f)
        #     
        #     if not data.get('updates') or not data['updates'].get('content') or len(data['updates']['content']) == 0:
        #         logger.info(f"Deleting {newest_file} - No updates found")
        #         os.remove(file_path)
        #         deleted_no_updates += 1
        #         
        # except Exception as e:
        #     logger.error(f"Error processing {newest_file}: {str(e)}")
    
    logger.info(f"\nCleanup Summary:")
    logger.info(f"Total JSON files processed: {total_files}")
    # logger.info(f"Files deleted (no updates): {deleted_no_updates}")
    logger.info(f"Files deleted (duplicates): {deleted_duplicates}")
    logger.info(f"Files remaining: {total_files - deleted_duplicates}")

if __name__ == "__main__":
    cleanup_no_update_files() 