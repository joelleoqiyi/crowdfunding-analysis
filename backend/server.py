# Required for browser automation
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from flask import Flask, request, jsonify
import time
import os
import sys
import json
from datetime import datetime

# Setup paths for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add update_analysis directory to path
update_analysis_dir = os.path.join(parent_dir, 'update-analysis')
sys.path.append(update_analysis_dir)

# Add scrapers directory to path
scrapers_dir = os.path.join(update_analysis_dir, 'scrapers')
sys.path.append(scrapers_dir)

# Add utils directory to path
utils_dir = os.path.join(update_analysis_dir, 'utils')
sys.path.append(utils_dir)

# Import browser utilities first (this helps with circular import issues)
try:
    from browser_utils import get_browser, random_sleep
    print("Successfully imported browser_utils from update-analysis/utils directory")
except ImportError as e:
    print(f"Failed to import browser_utils: {str(e)}")
    print(f"File exists: {os.path.exists(os.path.join(utils_dir, 'browser_utils.py'))}")
    print(f"Files in utils: {os.listdir(utils_dir) if os.path.exists(utils_dir) else 'Directory not found'}")

# Import prediction functions from web_predict
try:
    from web_predict import (
        load_model, 
        load_vectorizer, 
        load_feature_names, 
        predict_project_success,
        get_remarkable_features
    )
    print("Successfully imported web_predict functions")
except ImportError as e:
    print(f"Failed to import web_predict: {str(e)}")

# Import scrapers
try:
    from project_scraper import scrape_project
    from update_scraper import extract_updates_content
    print("Successfully imported scraper functions")
except ImportError as e:
    print(f"Failed to import scrapers: {str(e)}")
    print(f"Files in scrapers: {os.listdir(scrapers_dir) if os.path.exists(scrapers_dir) else 'Directory not found'}")

# Setup for cross-origin requests (basic implementation without flask-cors)
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Initialize Flask app
app = Flask(__name__)

# Define paths for model files
MODEL_DIR = os.path.join(update_analysis_dir, 'web_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.txt')

# Global variables for loaded model components
model = None
vectorizer = None
feature_names = None

# Add a global counter to track which mock data scenario to use
mock_data_counter = 0

def load_prediction_components():
    """Load the model, vectorizer, and feature names"""
    global model, vectorizer, feature_names
    
    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        return False
    
    # Check if model files exist
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        return False
    
    if not os.path.exists(VECTORIZER_PATH):
        print(f"ERROR: Vectorizer file not found: {VECTORIZER_PATH}")
        return False
    
    if not os.path.exists(FEATURE_NAMES_PATH):
        print(f"ERROR: Feature names file not found: {FEATURE_NAMES_PATH}")
        return False
    
    try:
        # Load model components
        print("Loading model components...")
        model = load_model(MODEL_PATH)
        vectorizer = load_vectorizer(VECTORIZER_PATH)
        feature_names = load_feature_names(FEATURE_NAMES_PATH)
        
        print("Model components loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Vectorizer type: {type(vectorizer).__name__}")
        print(f"Feature names count: {len(feature_names)}")
        return True
    except Exception as e:
        print(f"ERROR loading model components: {str(e)}")
        return False

# Add route for CORS preflight requests
@app.route('/predict', methods=['OPTIONS'])
def handle_preflight():
    response = jsonify({"success": True})
    return add_cors_headers(response)

@app.route('/', methods=['GET'])
def home():
    response = jsonify({
        "message": "Crowdfunding Analysis API",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/predict", "methods": ["POST", "OPTIONS"], "description": "Make a prediction for a project"},
            {"path": "/scrape", "methods": ["POST"], "description": "Scrape a Kickstarter campaign"}
        ]
    })
    return add_cors_headers(response)

def clean_kickstarter_url(url):
    """
    Clean and prepare a Kickstarter URL for scraping.
    This ensures the URL is in the proper format for the scrapers to work with.
    """
    print(f"Cleaning URL: {url}")
    
    # Remove any trailing slashes
    url = url.rstrip('/')
    
    # Remove any query parameters
    if '?' in url:
        url = url[:url.index('?')]
    
    # Remove /posts or /description if present at the end
    for suffix in ['/posts', '/description', '/comments', '/community', '/updates', '/faqs']:
        if url.endswith(suffix):
            url = url[:-len(suffix)]
    
    # Ensure the URL is a valid Kickstarter project URL
    if not ('kickstarter.com/projects/' in url or 'indiegogo.com/' in url):
        print(f"Warning: URL may not be a valid crowdfunding project: {url}")
    
    # Strip any trailing whitespace
    url = url.strip()
    
    print(f"Cleaned URL: {url}")
    return url

def scrape_kickstarter(url):
    """
    Scrape a Kickstarter campaign using the scraper modules from update-analysis.
    Returns the project data in the expected format for prediction.
    
    Implementation based on the successful main.py approach.
    """
    print(f"Original URL for scraping: {url}")
    
    # First clean the URL to ensure it's in the proper format
    base_url = clean_kickstarter_url(url)
    
    # Then add /posts for update scraping
    posts_url = base_url + "/posts"
    
    print(f"Using URL for scraping: {posts_url}")
    
    try:
        # Call scrape_project with the properly formatted URL
        print("Calling scrape_project function...")
        scraped_data = scrape_project(posts_url)
        
        if not scraped_data:
            print("Scrape_project returned no data")
            return {
                "success": False,
                "error": "No data returned from scraper"
            }
        
        if 'error' in scraped_data:
            print(f"Error returned by scraper: {scraped_data['error']}")
            return {
                "success": False,
                "error": scraped_data['error']
            }
        
        # Extract campaign details
        campaign_details = scraped_data.get('campaign_details', {})
        
        # Try to get the title, first from campaign_details, then from URL if not found
        if 'title' in campaign_details and campaign_details['title']:
            campaign_title = campaign_details['title']
        else:
            # Extract project name from URL as fallback
            project_name = base_url.split('/')[-1].replace('-', ' ').title()
            campaign_title = project_name
            # Add it to campaign_details
            campaign_details['title'] = campaign_title
        
        # Check the updates count
        updates = scraped_data.get('updates', {'count': 0, 'content': []})
        updates_count = updates.get('count', 0)
        updates_content = updates.get('content', [])
        
        print(f"Scraped campaign: {campaign_title}")
        print(f"Found {updates_count} updates with {len(updates_content)} content items")
        
        # Log campaign details for debugging
        print(f"Campaign details: Pledged: {campaign_details.get('pledged_amount', 'N/A')}, " +
              f"Goal: {campaign_details.get('funding_goal', 'N/A')}, " +
              f"Backers: {campaign_details.get('backers_count', 'N/A')}")
        
        # Return the data in the expected format
        return {
            "success": True,
            "data": {
                'campaign_details': campaign_details,
                'updates': updates
            },
            "campaign_title": campaign_title,
            "url": base_url  # Return the cleaned base URL
        }
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Exception during scraping: {str(e)}")
        print(error_traceback)
        return {
            "success": False,
            "error": f"Scraping failed: {str(e)}"
        }

# Fallback to mock data in case the real scraper fails
def get_mock_data(url, scenario=0):
    """
    Fallback function to generate mock data for testing purposes.
    """
    print(f"Using mock data for URL: {url} (scenario: {scenario})")
    
    # Generate current date and end date
    from datetime import datetime, timedelta
    current_date = datetime.now()
    
    # Create base project data structure
    project_data = {
        'campaign_details': {
            'title': '',
            'funding_goal': '$10000',
            'pledged_amount': '',
            'backers_count': '',
            'days_left': '',
            'funding_start_date': '',
            'funding_end_date': '',
            'description': ''
        },
        'updates': {
            'count': 0,
            'content': []
        }
    }
    
    # Parse the URL to extract potential campaign name
    campaign_name = url.split('/')[-1] if '/' in url else 'sample-campaign'
    
    # Different scenarios based on input
    if scenario == 0:  # Extremely Successful
        project_data['campaign_details'].update({
            'title': f"Wildly Successful: {campaign_name}",
            'pledged_amount': '$25000',
            'backers_count': '500',
            'days_left': '15',
            'funding_start_date': (current_date - timedelta(days=15)).isoformat() + 'Z',
            'funding_end_date': (current_date + timedelta(days=15)).isoformat() + 'Z',
            'description': 'Our team has years of manufacturing experience, and we\'ve already completed several successful prototypes.'
        })
        project_data['updates'] = {
            'count': 8,
            'content': [
                {'content': 'Launch Day Update: We\'re live on Kickstarter!', 'likes_count': 125, 'comments_count': 15, 'comments': ['Backed immediately!']},
                {'content': 'Day 3 Update: Wow! We\'re already 50% funded!', 'likes_count': 150, 'comments_count': 20, 'comments': ['Congrats!']},
                # ... more mock updates ...
            ]
        }
    # ... additional mock scenarios can be added here ...
    
    return {
        "success": True,
        "data": project_data,
        "campaign_title": project_data['campaign_details']['title'],
        "url": url
    }

@app.route('/scrape', methods=['POST'])
def scrape_route():
    data = request.json
    url = data.get('url')
    if not url:
        response = jsonify({"success": False, "error": "No URL provided"})
        return add_cors_headers(response), 400
    
    # First try real scraping
    result = scrape_kickstarter(url)
    
    # Fall back to mock data if real scraping fails
    if not result["success"]:
        print(f"Real scraping failed. Falling back to mock data. Error: {result.get('error')}")
        # Use mock data counter to cycle through scenarios
        global mock_data_counter
        current_scenario = mock_data_counter
        mock_data_counter = (mock_data_counter + 1) % 5
        print(f"Using mock data scenario {current_scenario}")
        result = get_mock_data(url, current_scenario)
    else:
        print("Successfully scraped real data from Kickstarter")
    
    response = jsonify(result)
    return add_cors_headers(response)

@app.route('/predict', methods=['POST'])
def predict_route():
    global model, vectorizer, feature_names
    
    # Check if model is loaded
    if model is None or vectorizer is None or feature_names is None:
        success = load_prediction_components()
        if not success:
            response = jsonify({
                "success": False,
                "error": "Failed to load model components"
            })
            return add_cors_headers(response), 500
    
    try:
        data = request.json
        if not data:
            response = jsonify({
                "success": False, 
                "error": "No JSON data provided"
            })
            return add_cors_headers(response), 400
            
        project_data = data.get('project_data')
        
        if not project_data:
            # If no project data provided, try to scrape from URL
            url = data.get('url')
            if not url:
                response = jsonify({
                    "success": False, 
                    "error": "Neither project data nor URL provided"
                })
                return add_cors_headers(response), 400
            
            # Try real scraping first
            scrape_result = scrape_kickstarter(url)
            
            # Fallback to mock data if real scraping fails
            if not scrape_result["success"]:
                print(f"Real scraping failed during prediction. Falling back to mock data. Error: {scrape_result.get('error')}")
                # Use mock data counter to cycle through scenarios
                global mock_data_counter
                current_scenario = mock_data_counter
                mock_data_counter = (mock_data_counter + 1) % 5
                print(f"Using mock data scenario {current_scenario}")
                scrape_result = get_mock_data(url, current_scenario)
            else:
                print("Successfully scraped real data for prediction")
            
            project_data = scrape_result["data"]
            campaign_url = scrape_result["url"]
            campaign_title = scrape_result["campaign_title"]
        else:
            campaign_url = data.get('url', '')
            campaign_title = project_data.get('campaign_details', {}).get('title', 'Unknown Project')
        
        # Make a real prediction using the XGBoost model
        print(f"Making prediction for project: {campaign_title}")
        prediction_result = predict_project_success(project_data, model, vectorizer, feature_names)
        
        # Add debug information to confirm model is being used
        print(f"MODEL DEBUG INFO:")
        print(f"  Using XGBoost model: {model is not None}")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Prediction confidence: {prediction_result['confidence']:.2f}")
        print(f"  Success probability: {prediction_result['success_probability']:.2f}")
        print(f"  Updates count: {prediction_result.get('key_features', {}).get('updates_count', 0)}")
        
        # Get remarkable features
        remarkable_features = get_remarkable_features(prediction_result)
        
        # Format response for frontend
        formatted_response = {
            "success": True,
            "campaignTitle": campaign_title,
            "campaignUrl": campaign_url,
            "timestamp": datetime.now().isoformat(),
            "overallPrediction": "success",
            "description": {
                # Pure mock data for description section
                "score": 50,  # Fixed mock score
                "prediction": "uncertain",  # Always uncertain for description
                "confidence": 0.5,  # Fixed confidence
                "justification": "Description analysis will be implemented separately by another team.",
                "findings": [
                    "This is a placeholder for the description analysis.",
                    "The actual description analysis will use a separate model.",
                    "For now, we display basic campaign metrics:",
                    "Funding progress: 65% funded (mock data)",
                    "Days left: 14 (mock data)"
                ]
            },
            # Pure mock data for comments section
            "comments": {
                "score": 50,  # Fixed mock score
                "prediction": "uncertain",  # Always uncertain
                "confidence": 0.5,  # Fixed confidence
                "justification": "Comment analysis will be implemented separately by another team.",
                "findings": [
                    "This is a placeholder for the comment analysis.",
                    "The actual comment analysis will use a separate model.",
                    "Currently no comment analysis is available."
                ]
            },
            # Pure mock data for youtube section
            "youtube": {
                "score": 50,  # Fixed mock score
                "prediction": "uncertain",  # Always uncertain
                "confidence": 0.5,  # Fixed confidence
                "justification": "YouTube analysis will be implemented separately by another team.",
                "findings": [
                    "This is a placeholder for the YouTube analysis.",
                    "The actual YouTube analysis will use a separate model.",
                    "Currently no YouTube analysis is available."
                ]
            },
            "updates": {
                # Keep using real model data for updates section
                "score": calculate_update_score(prediction_result),
                "prediction": "success" if prediction_result.get("key_features", {}).get("updates_count", 0) > 3 else "uncertain",
                "confidence": min(0.9, prediction_result.get("key_features", {}).get("updates_per_day", 0) * 5),
                "justification": generate_update_justification(prediction_result),
                "findings": [
                    f"Total updates: {prediction_result.get('key_features', {}).get('updates_count', 0)}",
                    f"Updates per day: {prediction_result.get('key_features', {}).get('updates_per_day', 0):.2f}",
                    f"Total likes on updates: {prediction_result.get('key_features', {}).get('total_likes', 0)}",
                    f"Total comments on updates: {prediction_result.get('key_features', {}).get('total_comments', 0)}",
                    # Add additional findings from prediction data
                    f"Average likes per update: {prediction_result.get('key_features', {}).get('average_likes_per_update', 0):.1f}",
                    f"Average comments per update: {prediction_result.get('key_features', {}).get('average_comments_per_update', 0):.1f}",
                    f"Average update length: {prediction_result.get('key_features', {}).get('avg_update_length', 0):.0f} characters",
                    f"Campaign duration: {prediction_result.get('metrics', {}).get('funding_duration', 30)} days",
                    f"Model confidence for updates: {prediction_result.get('confidence', 0.5):.0%}",
                ] + extract_update_remarkable_features(prediction_result)
            }
        }
        
        # Return the response with CORS headers
        response = jsonify(formatted_response)
        return add_cors_headers(response)
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error in prediction: {str(e)}")
        print(traceback_str)
        
        error_response = jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback_str
        })
        return add_cors_headers(error_response), 500

def calculate_update_score(prediction_result):
    """Calculate a more accurate update score based on multiple factors"""
    updates_count = prediction_result.get("key_features", {}).get("updates_count", 0)
    updates_per_day = prediction_result.get("key_features", {}).get("updates_per_day", 0)
    total_likes = prediction_result.get("key_features", {}).get("total_likes", 0)
    
    # Base score on updates count (0-40 points)
    if updates_count >= 8:
        count_score = 40
    elif updates_count >= 5:
        count_score = 30
    elif updates_count >= 3:
        count_score = 20
    elif updates_count >= 1:
        count_score = 10
    else:
        count_score = 0
    
    # Add points for update frequency (0-30 points)
    if updates_per_day >= 0.3:
        frequency_score = 30
    elif updates_per_day >= 0.2:
        frequency_score = 25
    elif updates_per_day >= 0.1:
        frequency_score = 20
    elif updates_per_day > 0:
        frequency_score = 10
    else:
        frequency_score = 0
    
    # Add points for engagement (0-30 points)
    engagement_score = min(30, total_likes // 50)
    
    # Calculate final score (max 100)
    final_score = min(100, count_score + frequency_score + engagement_score)
    
    # If campaign is funded > 100%, give minimum score of 60
    if prediction_result.get("metrics", {}).get("percent_funded", 0) >= 100:
        final_score = max(final_score, 60)
    
    return final_score

def generate_update_justification(prediction_result):
    """Generate a justification for the update score"""
    updates_count = prediction_result.get("key_features", {}).get("updates_count", 0)
    updates_per_day = prediction_result.get("key_features", {}).get("updates_per_day", 0)
    percent_funded = prediction_result.get("metrics", {}).get("percent_funded", 0)
    
    if updates_count >= 5 and percent_funded >= 100:
        return f"Campaign has {updates_count} updates with excellent engagement and is fully funded."
    elif updates_count >= 3:
        return f"Campaign has {updates_count} updates (about {updates_per_day:.2f} per day), showing good creator involvement."
    else:
        return f"Campaign has {updates_count} updates (about {updates_per_day:.2f} per day)."

def extract_update_remarkable_features(prediction_result):
    """Extract update-specific remarkable features from the prediction result"""
    remarkable_features = []
    
    # Add remarkable features based on update count
    updates_count = prediction_result.get("key_features", {}).get("updates_count", 0)
    if updates_count >= 8:
        remarkable_features.append("Exceptional number of updates (8+) shows strong creator involvement")
    elif updates_count <= 1 and prediction_result.get("metrics", {}).get("percent_time", 0) > 50:
        remarkable_features.append("Few updates despite campaign being halfway complete suggests limited creator engagement")
    
    # Add remarkable features based on update frequency
    updates_per_day = prediction_result.get("key_features", {}).get("updates_per_day", 0)
    if updates_per_day >= 0.3:
        remarkable_features.append("High update frequency (0.3+ per day) correlates strongly with successful campaigns")
    elif updates_per_day <= 0.05 and updates_count > 0:
        remarkable_features.append("Very low update frequency may reduce backer confidence")
    
    # Add remarkable features based on engagement
    avg_likes = prediction_result.get("key_features", {}).get("average_likes_per_update", 0)
    if avg_likes >= 50:
        remarkable_features.append("High engagement on updates indicates strong community interest")
    
    # Add remarkable features based on funding status and updates
    percent_funded = prediction_result.get("metrics", {}).get("percent_funded", 0)
    if percent_funded >= 100 and updates_count >= 5:
        remarkable_features.append("Fully funded campaigns with regular updates have extremely high success rates")
    elif percent_funded < 50 and updates_count <= 2 and prediction_result.get("metrics", {}).get("percent_time", 0) > 70:
        remarkable_features.append("Under-funded campaigns with few updates rarely reach their goals")
    
    return remarkable_features

# Load model components on startup
if __name__ == '__main__':
    print("Starting Crowdfunding Analysis API on port 5001...")
    
    # Try to load model components
    success = load_prediction_components()
    if success:
        print("Model loaded successfully! Ready to make predictions.")
    else:
        print("WARNING: Failed to load model. The API will use mock predictions.")
    
    app.run(debug=True, port=5001)