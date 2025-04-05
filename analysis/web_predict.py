import os
import pickle
import numpy as np
import re
from datetime import datetime, timezone, timedelta

def load_model(model_path):
    """Load the pre-trained XGBoost model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_vectorizer(vectorizer_path):
    """Load the pre-trained TF-IDF vectorizer."""
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

def load_feature_names(feature_names_path):
    """Load the feature names used by the model."""
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f]
    return feature_names

def _clean_money_value(value):
    """Convert a monetary string or value to a float."""
    if isinstance(value, str):
        # Remove currency symbols, commas, and convert to float
        return float(re.sub(r'[^\d.]', '', value.replace(',', '')) or 0)
    return float(value)

def _calculate_time_metrics(data, campaign):
    """Calculate time-based metrics for a project."""
    try:
        percent_funded = data['percent_funded']
        
        if 'funding_start_date' in campaign and 'funding_end_date' in campaign:
            # Parse dates and calculate time progression
            start_date = datetime.fromisoformat(campaign['funding_start_date'].replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(campaign['funding_end_date'].replace('Z', '+00:00'))
            total_duration = (end_date - start_date).total_seconds()
            
            # Calculate current position in timeline
            current_date = datetime.now(timezone.utc)
            if 'days_left' in campaign:
                days_left = int(campaign['days_left'])
                current_date = end_date - timedelta(days=days_left)
            
            elapsed = (current_date - start_date).total_seconds()
            percent_time = (elapsed / total_duration * 100) if total_duration > 0 else 0
            data['percent_time'] = percent_time
            
            # Project final funding percentage
            if percent_time > 0:
                projected_final = (percent_funded / percent_time) * 100
                data['projected_final_percent'] = projected_final
            else:
                # For very new projects, use a simple threshold
                data['success'] = 0 if percent_funded < 10 else 1
        else:
            # Fallback when time data is missing
            data['success'] = 1 if percent_funded >= 50 else 0
    except Exception as e:
        print(f"Error calculating time metrics: {str(e)}")
        data['success'] = 1 if data['percent_funded'] >= 50 else 0
    
    return data

def extract_features(project):
    """Extract relevant features from project data."""
    features = {}
    campaign = project.get('campaign_details', {})
    
    # Basic update statistics
    updates = project.get('updates', {})
    features['num_updates'] = num_updates = updates.get('count', 0)
    
    # Update content analysis
    update_contents = []
    total_likes = 0
    total_comments = 0
    total_comment_count = 0
    update_length_total = 0
    
    # Process each update
    for update in updates.get('content', []):
        update_content = update.get('content', '')
        update_contents.append(update_content)
        
        # Engagement metrics
        total_likes += update.get('likes_count', 0)
        total_comments += len(update.get('comments', []))
        total_comment_count += update.get('comments_count', 0)
        
        # Update length calculations
        if update_content:
            update_length_total += len(update_content)
    
    # Store counting metrics
    features['total_likes'] = total_likes
    features['total_comments'] = total_comments
    features['total_comment_count'] = total_comment_count
    
    # Calculate averages (avoid division by zero)
    features['avg_update_length'] = update_length_total / max(num_updates, 1)
    features['average_likes_per_update'] = total_likes / max(num_updates, 1)
    features['average_comments_per_update'] = total_comments / max(num_updates, 1)
    
    # Set placeholder values for early update indicators
    features['has_first_day_update'] = 0  # Placeholder for future implementation
    features['has_first_week_update'] = 0  # Placeholder for future implementation
    
    # Campaign duration and timing
    funding_duration = 30  # Default value
    if 'funding_start_date' in campaign and 'funding_end_date' in campaign:
        try:
            start_date = datetime.fromisoformat(campaign['funding_start_date'].replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(campaign['funding_end_date'].replace('Z', '+00:00'))
            funding_duration = (end_date - start_date).days
        except:
            pass
    
    features['funding_duration'] = funding_duration
    
    # Calculate update frequency
    features['updates_per_day'] = features['num_updates'] / max(funding_duration, 1)
    
    # Extract backers count and calculate related metrics
    backers_count = 0
    if 'backers_count' in campaign:
        try:
            backers = campaign['backers_count']
            if isinstance(backers, str):
                backers_count = float(re.sub(r'[^\d.]', '', backers))
            else:
                backers_count = float(backers)
        except:
            pass
    
    features['backers_count'] = backers_count
    
    # Add time-based metrics
    features['days_left'] = int(campaign.get('days_left', 0))
    features['percent_time'] = project.get('percent_time', 0)
    
    # Calculate backer acquisition rate
    time_denominator = max(funding_duration * (features['percent_time']/100), 1)
    features['backers_per_day'] = backers_count / time_denominator
    
    # Calculate average pledge amount
    if backers_count > 0:
        features['avg_pledge_amount'] = project.get('clean_pledged_amount', 0) / backers_count
    else:
        features['avg_pledge_amount'] = 0
    
    # Store concatenated update text for later analysis
    features['all_updates_text'] = ' '.join(update_contents)
    
    return features

def prepare_project_data(project_json):
    """Prepare a project's data from scraped JSON for prediction."""
    # Create a project object with the required fields
    project = {
        'campaign_details': project_json.get('campaign_details', {})
    }
    
    # Clean monetary values
    funding_goal = project['campaign_details'].get('funding_goal', '0')
    pledged_amount = project['campaign_details'].get('pledged_amount', '0')
    
    project['clean_funding_goal'] = _clean_money_value(funding_goal)
    project['clean_pledged_amount'] = _clean_money_value(pledged_amount)
    
    # Calculate percentage funded
    percent_funded = (project['clean_pledged_amount'] / project['clean_funding_goal'] * 100) if project['clean_funding_goal'] > 0 else 0
    project['percent_funded'] = percent_funded
    
    # Add updates data
    project['updates'] = project_json.get('updates', {})
    
    # Calculate time metrics
    project = _calculate_time_metrics(project, project['campaign_details'])
    
    return project

def predict_project_success(project_json, model, vectorizer, feature_names):
    """Predict whether a project will succeed based on JSON data."""
    # Prepare project data
    project = prepare_project_data(project_json)
    
    # Extract features
    features = extract_features(project)
    
    # Prepare feature vector matching the training data structure
    feature_vector = []
    for feature in feature_names:
        if feature.startswith('tfidf_'):
            continue  # Skip TF-IDF features here, they'll be added later
        feature_vector.append(features.get(feature, 0))
    
    # Convert to numpy array
    X_numerical = np.array(feature_vector).reshape(1, -1)
    
    # Get text features from vectorizer
    text_data = features.get('all_updates_text', '')
    X_text = vectorizer.transform([text_data]).toarray()
    
    # Combine numerical and text features
    X = np.hstack([X_numerical, X_text])
    
    # Make prediction
    prediction_proba = model.predict_proba(X)[0]
    predicted_class = model.predict(X)[0]
    
    # Extract key metrics for the response
    percent_funded = project.get('percent_funded', 0)
    percent_time = project.get('percent_time', 0)
    
    # Calculate funding trajectory
    funding_trajectory = (percent_funded / percent_time) if percent_time > 0 else 0
    projected_final = funding_trajectory * 100 if percent_time > 0 else 0
    
    # Format result
    result = {
        'success_prediction': bool(predicted_class),
        'success_probability': float(prediction_proba[1]),  # Probability of success
        'prediction_label': 'Likely to succeed' if predicted_class else 'Likely to fail',
        'confidence': float(max(prediction_proba)),
        'metrics': {
            'percent_funded': float(percent_funded),
            'percent_time': float(percent_time),
            'funding_trajectory': float(funding_trajectory),
            'projected_final_percent': float(projected_final),
            'days_left': int(project['campaign_details'].get('days_left', 0)),
            'backers_count': int(features.get('backers_count', 0)),
            'funding_duration': int(features.get('funding_duration', 30))
        },
        'key_features': {
            'updates_count': int(features.get('num_updates', 0)),
            'total_likes': int(features.get('total_likes', 0)),
            'total_comments': int(features.get('total_comments', 0)),
            'avg_update_length': float(features.get('avg_update_length', 0)),
            'updates_per_day': float(features.get('updates_per_day', 0)),
            'avg_pledge_amount': float(features.get('avg_pledge_amount', 0))
        },
        'remarkable_features': get_remarkable_features(features)
    }
    
    return result

def get_remarkable_features(features):
    """Identify remarkable features of a project."""
    remarkable = []
    
    # Define thresholds for remarkable features
    thresholds = {
        'num_updates': 5,
        'updates_per_day': 0.3,
        'backers_count': 100,
        'backers_per_day': 10,
        'total_likes': 50,
        'total_comments': 20,
        'average_likes_per_update': 10,
        'average_comments_per_update': 5,
        'avg_pledge_amount': 50,
    }
    
    # Check each metric against thresholds
    for metric, threshold in thresholds.items():
        if features.get(metric, 0) >= threshold:
            readable_metric = {
                'num_updates': 'high update count',
                'updates_per_day': 'frequent updates',
                'backers_count': 'many backers',
                'backers_per_day': 'fast backer growth',
                'total_likes': 'highly liked',
                'total_comments': 'heavily commented',
                'average_likes_per_update': 'engaging updates',
                'average_comments_per_update': 'interactive community',
                'avg_pledge_amount': 'high average pledge'
            }.get(metric, metric)
            
            remarkable.append(readable_metric)
    
    return remarkable

def example_usage():
    """Demonstrate how to use the prediction functionality."""
    # Define paths to model files
    model_dir = 'crowdfunding-analysis/analysis/web_model'
    model_path = os.path.join(model_dir, 'xgboost_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    feature_names_path = os.path.join(model_dir, 'feature_names.txt')
    
    # Load model, vectorizer, and feature names
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)
    feature_names = load_feature_names(feature_names_path)
    
    # Example project data (this would come from your web scraper)
    example_project = {
        'campaign_details': {
            'funding_goal': '$10000',
            'pledged_amount': '$5000',
            'days_left': 15,
            'funding_start_date': '2025-03-01T00:00:00Z',
            'funding_end_date': '2025-04-01T00:00:00Z',
            'backers_count': 50
        },
        'updates': {
            'count': 3,
            'content': [
                {
                    'content': "Thank you all for your support! We're making great progress!",
                    'likes_count': 20,
                    'comments': ['Great job!', 'Looking forward to it!'],
                    'comments_count': 2
                },
                {
                    'content': "Here's an update on our manufacturing process.",
                    'likes_count': 15,
                    'comments': ['Nice!'],
                    'comments_count': 1
                },
                {
                    'content': "We've just reached 50% of our funding goal!",
                    'likes_count': 25,
                    'comments': ['Congrats!', 'Awesome!'],
                    'comments_count': 2
                }
            ]
        }
    }
    
    # Make prediction
    prediction = predict_project_success(example_project, model, vectorizer, feature_names)
    
    # Print result
    print("Project Prediction:")
    print(f"- Prediction: {prediction['prediction_label']}")
    print(f"- Confidence: {prediction['confidence']:.1%}")
    print(f"- Success Probability: {prediction['success_probability']:.1%}")
    
    print("\nKey Metrics:")
    print(f"- Funding: {prediction['metrics']['percent_funded']:.1f}% funded")
    print(f"- Time: {prediction['metrics']['percent_time']:.1f}% elapsed")
    print(f"- Trajectory: {prediction['metrics']['funding_trajectory']:.2f}x (>1.0 is good)")
    print(f"- Projected Final: {prediction['metrics']['projected_final_percent']:.1f}%")
    
    print("\nRemarkable Features:")
    if prediction['remarkable_features']:
        for feature in prediction['remarkable_features']:
            print(f"- {feature}")
    else:
        print("- None identified")

if __name__ == "__main__":
    example_usage() 