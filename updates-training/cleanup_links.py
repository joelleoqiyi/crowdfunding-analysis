import os

def cleanup_links_file():
    """
    Removes duplicate links from unscraped_links.txt and ensures proper format with update count
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    links_file = 'unscraped_links.txt'
    
    if not os.path.exists(links_file):
        print(f"File {links_file} not found")
        return
    
    # Read all links
    with open(links_file, 'r') as f:
        lines = f.readlines()
    
    # Process links to remove duplicates
    unique_links = set()
    cleaned_links = []
    
    for line in lines:
        if not line.strip():
            continue
            
        # Split by tab and take just the URL
        parts = line.strip().split('\t')
        url = parts[0]
        
        if url not in unique_links:
            unique_links.add(url)
            cleaned_links.append(url)
    
    # Write back the cleaned links with tab and update count
    with open(links_file, 'w') as f:
        for link in cleaned_links:
            f.write(f"{link}\t0\n")
    
    print(f"Cleaned up {links_file}")
    print(f"Original line count: {len(lines)}")
    print(f"Unique links count: {len(cleaned_links)}")

if __name__ == "__main__":
    cleanup_links_file() 