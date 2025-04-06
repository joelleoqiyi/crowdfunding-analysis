# Remove or comment out selenium since we're using mock data
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options

from flask import Flask, request, jsonify
import time
import os
import sys
import json
from datetime import datetime

# Add path to update_analysis for model import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
update_analysis_dir = os.path.join(parent_dir, 'update-analysis')
sys.path.append(update_analysis_dir)

# Import prediction functions from web_predict
from web_predict import (
    load_model, 
    load_vectorizer, 
    load_feature_names, 
    predict_project_success,
    get_remarkable_features
)

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

def scrape_kickstarter(url):
    """
    Mock scraper function that returns sample data from 5 different campaign scenarios.
    The function cycles through the scenarios with each call.
    In a production environment, you would use Selenium to scrape the actual data.
    """
    global mock_data_counter
    
    print(f"Mock scraping URL: {url}")
    
    try:
        # Parse the URL to extract potential campaign name
        campaign_name = url.split('/')[-1] if '/' in url else 'sample-campaign'
        
        # Generate current date and end date
        from datetime import datetime, timedelta
        current_date = datetime.now()
        
        # Increment counter and wrap around to cycle through scenarios
        mock_data_counter = (mock_data_counter + 1) % 5
        scenario = mock_data_counter
        
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
        
        # Scenario 1: Extremely Successful Campaign (250% funded, many updates, strong engagement)
        if scenario == 0:
            project_data['campaign_details'].update({
                'title': f"Wildly Successful: {campaign_name}",
                'pledged_amount': '$25000',   # 250% funded
                'backers_count': '500',       # Strong backer count
                'days_left': '15',            # Plenty of time left
                'funding_start_date': (current_date - timedelta(days=15)).isoformat() + 'Z',
                'funding_end_date': (current_date + timedelta(days=15)).isoformat() + 'Z',
                'description': 'Our team has years of manufacturing experience, and we\'ve already completed several successful prototypes. We have established partnerships with reliable suppliers and a clear production timeline. The funds will help us scale production and deliver on time.'
            })
            project_data['updates'] = {
                'count': 8,  # Many updates
                'content': [
                    {'content': 'Launch Day Update: We\'re live on Kickstarter!', 'likes_count': 125, 'comments_count': 15, 'comments': ['Backed immediately!', 'Looks amazing!']},
                    {'content': 'Day 3 Update: Wow! We\'re already 50% funded!', 'likes_count': 150, 'comments_count': 20, 'comments': ['Congrats!', 'This is going to blow past the goal!']},
                    {'content': 'Week 1 Recap: We\'re now FULLY FUNDED!', 'likes_count': 200, 'comments_count': 25, 'comments': ['Amazing news!', 'Well deserved!']},
                    {'content': 'Production Update: Manufacturing partners confirmed', 'likes_count': 175, 'comments_count': 18, 'comments': ['Love seeing the process!']},
                    {'content': 'Stretch Goal UNLOCKED! 200% funding reached', 'likes_count': 210, 'comments_count': 30, 'comments': ['You guys rock!']},
                    {'content': 'Meet the Team: Background of our founders', 'likes_count': 185, 'comments_count': 22, 'comments': ['Impressive team!']},
                    {'content': 'Manufacturing Timeline: Detailed schedule', 'likes_count': 160, 'comments_count': 15, 'comments': ['Very professional approach']},
                    {'content': 'Final Week! 250% funded - new stretch goals', 'likes_count': 230, 'comments_count': 35, 'comments': ['Take my money!']},
                ]
            }
            print("Using WILDLY SUCCESSFUL campaign scenario")
            
        # Scenario 2: Moderately Successful (120% funded, good updates)
        elif scenario == 1:
            project_data['campaign_details'].update({
                'title': f"Successful Campaign: {campaign_name}",
                'pledged_amount': '$12000',   # 120% funded
                'backers_count': '180',       # Decent backer count
                'days_left': '10',            # Some time left
                'funding_start_date': (current_date - timedelta(days=20)).isoformat() + 'Z',
                'funding_end_date': (current_date + timedelta(days=10)).isoformat() + 'Z',
                'description': 'We\'ve been working on this project for over a year. Our goal is to create a quality product that solves a real problem. We have the expertise to deliver and are committed to maintaining open communication.'
            })
            project_data['updates'] = {
                'count': 5,  # Good number of updates
                'content': [
                    {'content': 'Welcome to our campaign! Thanks for your support', 'likes_count': 75, 'comments_count': 10, 'comments': ['Excited for this!']},
                    {'content': 'First week update: 40% funded!', 'likes_count': 82, 'comments_count': 12, 'comments': ['Great progress!']},
                    {'content': 'Halfway point update: 60% funded!', 'likes_count': 90, 'comments_count': 15, 'comments': ['Keep it up!']},
                    {'content': 'Production plans and timeline', 'likes_count': 85, 'comments_count': 13, 'comments': ['Thanks for the details']},
                    {'content': 'We\'re funded! What\'s next for the project', 'likes_count': 120, 'comments_count': 25, 'comments': ['Congratulations!']},
                ]
            }
            print("Using MODERATELY SUCCESSFUL campaign scenario")
            
        # Scenario 3: Borderline/Ambiguous (55% funded, medium updates)
        elif scenario == 2:
            project_data['campaign_details'].update({
                'title': f"Ambiguous Campaign: {campaign_name}",
                'pledged_amount': '$5500',    # 55% funded
                'backers_count': '85',        # Modest backer count
                'days_left': '5',             # Not much time left
                'funding_start_date': (current_date - timedelta(days=25)).isoformat() + 'Z',
                'funding_end_date': (current_date + timedelta(days=5)).isoformat() + 'Z',
                'description': 'Our innovative product addresses a gap in the market. While this is our first crowdfunding campaign, we bring industry experience and a clear vision. We hope you\'ll join us on this journey.'
            })
            project_data['updates'] = {
                'count': 3,  # Few updates
                'content': [
                    {'content': 'Campaign launch! Thank you for checking us out', 'likes_count': 45, 'comments_count': 7, 'comments': ['Interesting concept']},
                    {'content': 'Progress update: 30% funded after two weeks', 'likes_count': 40, 'comments_count': 5, 'comments': ['Hope you make it!']},
                    {'content': 'New product features and development update', 'likes_count': 52, 'comments_count': 8, 'comments': ['Looking forward to this']},
                ]
            }
            print("Using AMBIGUOUS campaign scenario")
            
        # Scenario 4: Likely to Fail (30% funded, few updates)
        elif scenario == 3:
            project_data['campaign_details'].update({
                'title': f"Struggling Campaign: {campaign_name}",
                'pledged_amount': '$3000',    # 30% funded
                'backers_count': '42',        # Low backer count
                'days_left': '2',             # Almost out of time
                'funding_start_date': (current_date - timedelta(days=28)).isoformat() + 'Z',
                'funding_end_date': (current_date + timedelta(days=2)).isoformat() + 'Z',
                'description': 'We\'re excited to bring this product to market. While we don\'t have manufacturing experience, we believe we can figure it out as we go. The timeline might be ambitious, but we\'re optimistic.'
            })
            project_data['updates'] = {
                'count': 2,  # Very few updates
                'content': [
                    {'content': 'Launch day! Our campaign is live', 'likes_count': 20, 'comments_count': 4, 'comments': ['Good luck!']},
                    {'content': 'Week 2 update: Production challenges', 'likes_count': 15, 'comments_count': 3, 'comments': ['Hope you overcome these issues']},
                ]
            }
            print("Using STRUGGLING campaign scenario")
            
        # Scenario 5: Failing Badly (15% funded, minimal engagement)
        else:
            project_data['campaign_details'].update({
                'title': f"Failing Campaign: {campaign_name}",
                'pledged_amount': '$1500',    # 15% funded
                'backers_count': '20',        # Very few backers
                'days_left': '3',             # Almost no time left
                'funding_start_date': (current_date - timedelta(days=27)).isoformat() + 'Z',
                'funding_end_date': (current_date + timedelta(days=3)).isoformat() + 'Z',
                'description': 'This is our first attempt at creating a product. We have a basic prototype but are still working on many details. We hope to use the funds to figure out manufacturing and delivery.'
            })
            project_data['updates'] = {
                'count': 1,  # Single update
                'content': [
                    {'content': 'Welcome to our campaign!', 'likes_count': 12, 'comments_count': 2, 'comments': ['What\'s your timeline?', 'Have you made prototypes?']},
                ]
            }
            print("Using FAILING campaign scenario")
        
        return {
            "success": True,
            "data": project_data,
            "campaign_title": project_data['campaign_details']['title'],
            "url": url
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error in mock scraper: {str(e)}"
        }

@app.route('/scrape', methods=['POST'])
def scrape_route():
    data = request.json
    url = data.get('url')
    if not url:
        response = jsonify({"success": False, "error": "No URL provided"})
        return add_cors_headers(response), 400
    
    result = scrape_kickstarter(url)
    if not result["success"]:
        response = jsonify(result)
        return add_cors_headers(response), 500
    
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
            
            scrape_result = scrape_kickstarter(url)
            if not scrape_result["success"]:
                response = jsonify(scrape_result)
                return add_cors_headers(response), 500
            
            project_data = scrape_result["data"]
            campaign_url = url
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