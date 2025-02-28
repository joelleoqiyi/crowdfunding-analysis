def discover_link_to_url(url: str):
    return url.split('?')[0]

def get_comments_link(url: str):
    return url + "/comments"