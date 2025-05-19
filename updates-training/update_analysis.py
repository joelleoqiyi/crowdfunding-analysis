import json
import os
import re
from datetime import datetime, timezone, timedelta
from io import StringIO
from pathlib import Path

# Data handling
import pandas as pd
import numpy as np

# Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, roc_auc_score
)
from sklearn.utils import class_weight

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Constants and configuration
COLORS = {
    'primary': '#1f77b4',    # Main color for plots
    'success': '#2ca02c',    # Color for successful projects
    'failure': '#d62728',    # Color for failed projects
    'neutral': '#7f7f7f',    # Neutral color
    'highlight': '#ff7f0e',  # Highlighting important elements
    'background': '#f5f5f5'  # Background color
}

# Feature labels for readability
FEATURE_LABELS = {
    'funding_duration': 'Campaign Duration',
    'total_likes': 'Total Likes',
    'total_comments': 'Total Comments',
    'total_comment_count': 'Comment Count',
    'num_updates': 'Number of Updates',
    'updates_per_day': 'Updates per Day',
    'avg_update_length': 'Average Update Length',
    'percent_time': 'Percentage of Time Elapsed',
    'backers_count': 'Number of Backers',
    'backers_per_day': 'Backers per Day',
    'avg_pledge_amount': 'Average Pledge Amount',
    'average_likes_per_update': 'Average Likes per Update',
    'average_comments_per_update': 'Average Comments per Update',
    'days_left': 'Days Left in Campaign',
    'has_first_day_update': 'Has Update on First Day',
    'has_first_week_update': 'Has Update in First Week'
}

# Optional pydotplus for visualization
pydotplus_available = False
try:
    import pydotplus
    import matplotlib.colors as mcolors
    pydotplus_available = True
except ImportError:
    print("pydotplus not available. Decision tree visualization will be disabled.")
    class DummyColors:
        def __getattr__(self, name):
            return '#000000'  # Default color
    mcolors = DummyColors()

def load_project_data(data_dir):
    """Load project data from JSON files.
    
    Args:
        data_dir (str): Directory containing project JSON files
        
    Returns:
        list: Processed project data objects
    """
    projects = []
    
    # Resolve data directory path
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' not found")
        # Try to find the correct path
        script_dir = Path(__file__).parent.absolute()
        parent_dir = script_dir.parent
        alternatives = [
            parent_dir / "webscraper" / "scrapers" / "scraped_data",
            script_dir / "webscraper" / "scrapers" / "scraped_data"
        ]
        
        for alt_path in alternatives:
            if alt_path.exists():
                print(f"Found data directory at: {alt_path}")
                data_path = alt_path
                break
        else:
            print("Could not locate scraped_data directory. Please check the path.")
            return []
    
    print(f"Loading data from: {data_path}")
    
    # Process all JSON files in the directory
    file_count = 0
    json_files = list(data_path.glob("*.json"))
    
    for json_file in json_files:
        file_count += 1
        try:
            # Load and validate the project data
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Skip if no campaign details
                if 'campaign_details' not in data or not data['campaign_details']:
                    print(f"Skipping {json_file.name} - no campaign details")
                    continue
                
                campaign = data['campaign_details']
    
                # Only process live projects (those with days_left field)
                if 'days_left' not in campaign:
                    continue
                
                # Extract project name from URL for better identification
                if 'url' in data:
                    data['title'] = data['url'].split('/')[-1]
                
                # Clean and normalize monetary values
                data['clean_funding_goal'] = _clean_money_value(campaign.get('funding_goal', '0'))
                data['clean_pledged_amount'] = _clean_money_value(campaign.get('pledged_amount', '0'))
                data['is_live'] = True
                
                # Calculate funding percentage
                funding_goal = data['clean_funding_goal']
                pledged_amount = data['clean_pledged_amount']
                percent_funded = (pledged_amount / funding_goal * 100) if funding_goal > 0 else 0
                data['percent_funded'] = percent_funded
                
                # Calculate time metrics and success prediction
                _calculate_time_metrics(data, campaign)
                
                projects.append(data)
                
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {json_file.name}, skipping file")
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}, skipping file")
    
    print(f"Found {file_count} JSON files, loaded {len(projects)} live projects successfully")
    return projects

def _clean_money_value(value):
    """Convert a monetary string or value to a float."""
    if isinstance(value, str):
        # Remove currency symbols, commas, and convert to float
        return float(re.sub(r'[^\d.]', '', value.replace(',', '')) or 0)
    return float(value)

def _calculate_time_metrics(data, campaign):
    """Calculate time-based metrics and success prediction for a project."""
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
                
                # Predict success based on trajectory
                if projected_final >= 100:
                    data['success'] = 1
                    data['success_prediction'] = f"{percent_funded:.1f}% funded at {percent_time:.1f}% of time - On track to succeed"
                else:
                    data['success'] = 0
                    data['success_prediction'] = f"{percent_funded:.1f}% funded at {percent_time:.1f}% of time - Projected to reach {projected_final:.1f}%"
            else:
                # For very new projects, use a simple threshold
                data['success'] = 0 if percent_funded < 10 else 1
        else:
            # Fallback when time data is missing
            data['success'] = 1 if percent_funded >= 50 else 0
            data['success_prediction'] = f"{percent_funded:.1f}% funded - insufficient date data"
    except Exception as e:
        print(f"Error calculating time metrics: {str(e)}")
        data['success'] = 1 if data['percent_funded'] >= 50 else 0

def extract_features(project, for_training=True):
    """Extract relevant features from project data.
    
    Args:
        project (dict): Project data dictionary
        for_training (bool): If True, extract features for model training
        
    Returns:
        dict: Feature dictionary
    """
    features = {}
    campaign = project.get('campaign_details', {})
    
    # Extract update metrics
    features.update(_extract_update_metrics(project))
    
    # Extract campaign metrics
    features.update(_extract_campaign_metrics(project, campaign))
    
    # Extract specialized features for training
    if for_training:
        features.update(_extract_training_features(project, campaign, features))
    
    return features

def _extract_update_metrics(project):
    """Extract metrics related to project updates."""
    features = {}
    updates = project.get('updates', {})
    update_contents = []
    
    # Basic update count
    features['num_updates'] = num_updates = updates.get('count', 0)
    
    # Initialize counters
    total_likes = 0
    total_comments = 0
    total_comment_count = 0
    update_length_total = 0
    
    # Process each update
    for update in updates.get('content', []):
        update_content = update.get('content', '')
        update_contents.append(update_content)
        
        # Accumulate engagement metrics
        total_likes += update.get('likes_count', 0)
        total_comments += len(update.get('comments', []))
        total_comment_count += update.get('comments_count', 0)
        
        # Measure content length
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
    
    # Create early update indicators
    features['has_first_day_update'] = 0  # Placeholder for future implementation
    features['has_first_week_update'] = 0  # Placeholder for future implementation
    
    # Store concatenated update text for later analysis
    features['all_updates_text'] = ' '.join(update_contents)
    
    return features

def _extract_campaign_metrics(project, campaign):
    """Extract metrics related to campaign configuration."""
    features = {}
    
    # Determine campaign duration
    funding_duration = _get_campaign_duration(campaign)
    features['funding_duration'] = funding_duration
    
    # Calculate update frequency
    if funding_duration > 0 and 'num_updates' in features:
        features['updates_per_day'] = features['num_updates'] / funding_duration
    else:
        features['updates_per_day'] = 0
    
    # Extract backers count
    features['backers_count'] = _get_backers_count(campaign)
    
    return features

def _get_campaign_duration(campaign):
    """Calculate or extract campaign duration in days."""
    # Use explicit duration if available
    if 'funding_duration_days' in campaign:
        return campaign.get('funding_duration_days', 0)
    
    # Calculate from start/end dates
    if 'funding_start_date' in campaign and 'funding_end_date' in campaign:
        try:
            start_date = datetime.fromisoformat(campaign['funding_start_date'].replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(campaign['funding_end_date'].replace('Z', '+00:00'))
            return (end_date - start_date).days
        except:
            pass
    
    # Default assumption
    return 30

def _get_backers_count(campaign):
    """Extract and clean backers count."""
    if 'backers_count' in campaign:
        try:
            backers = campaign['backers_count']
            if isinstance(backers, str):
                return float(re.sub(r'[^\d.]', '', backers))
            return float(backers)
        except:
            pass
    return 0

def _extract_training_features(project, campaign, base_features):
    """Extract additional features specifically for model training."""
    features = {}
    
    # Calculate backer acquisition rate (if appropriate features exist)
    funding_duration = base_features.get('funding_duration', 30)
    backers_count = base_features.get('backers_count', 0)
    percent_time = project.get('percent_time', 100)
    
    # Avoid division by zero with sensible defaults
    time_denominator = max(funding_duration * (percent_time/100), 1)
    features['backers_per_day'] = backers_count / time_denominator
    
    # Calculate average pledge amount (quality of backers)
    if backers_count > 0:
        features['avg_pledge_amount'] = project.get('clean_pledged_amount', 0) / backers_count
    else:
        features['avg_pledge_amount'] = 0
    
    # Add time-based metrics
    features['days_left'] = int(campaign.get('days_left', 0))
    features['percent_time'] = project.get('percent_time', 0)
    
    return features

def create_dataset(projects, for_training=True):
    """Create a dataset from the live project features."""
    features_list = []
    labels = []
    live_projects = []
    
    for project in projects:
        features = extract_features(project, for_training=for_training)
        live_projects.append((features, project))
        features_list.append(features)
        labels.append(project['success'])
    
    # Print dataset statistics
    print(f"Dataset created with {len(features_list)} live projects")
    print(f"Class distribution: {sum(labels)} projected to succeed, {len(labels) - sum(labels)} projected to fail")
    
    return features_list, labels, live_projects

def create_results_dir():
    """Create a fixed results directory that overwrites previous results."""
    # Use a fixed directory name instead of timestamp
    results_dir = 'crowdfunding-analysis/analysis/results'
    
    # Ensure directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Return the fixed directory path
    return results_dir

def create_readable_feature_importance(importance_df, tfidf_vectorizer=None):
    """Create a more readable version of feature importance rankings.
    
    Returns a list of tuples (feature_name, importance_value, description)
    """
    # Sort by importance
    if 'importance' in importance_df.columns:
        importance_df = importance_df.sort_values('importance', ascending=False).copy()
    
    result = []
    
    # Process each feature
    for _, row in importance_df.iterrows():
        feature = row['feature']
        importance = row['importance'] if 'importance' in row else 0
        
        # Get description based on feature type
        if feature in FEATURE_LABELS:
            description = FEATURE_LABELS[feature]
        elif feature.startswith('tfidf_') and tfidf_vectorizer is not None:
            # Try to get the word from the vectorizer
            try:
                idx = int(feature.split('_')[1])
                features_by_idx = {idx: word for word, idx in tfidf_vectorizer.vocabulary_.items()}
                word = features_by_idx.get(idx, f"Word {idx}")
                description = f"Word '{word}' in Updates"
            except:
                description = feature
        else:
            description = feature.replace('_', ' ').title()
        
        result.append((feature, importance, description))
    
    return result

def save_analysis_summary(results_dir, feature_importance, classification_report, tfidf_vectorizer,
                          live_projects_analysis, cv_scores=None, test_accuracy=None, f1_score=None,
                          roc_auc=None, class_weights=None, projects_summary=None):
    """Save a detailed summary of the analysis to a text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    file_path = os.path.join(results_dir, "analysis_summary.txt")
    
    with open(file_path, 'w') as f:
        f.write(f"=== Crowdfunding Project Analysis ===\n")
        f.write(f"Generated: {timestamp}\n\n")
        
        # Project statistics
        if projects_summary:
            f.write(f"Project Statistics:\n")
            f.write(f"- Total projects: {projects_summary.get('total', 'N/A')}\n")
            f.write(f"- Live projects: {projects_summary.get('live', 'N/A')}\n")
            f.write(f"- Successful projects: {projects_summary.get('successful', 'N/A')}\n")
            f.write(f"- Failed projects: {projects_summary.get('failed', 'N/A')}\n\n")
        
        # Model performance metrics
        if cv_scores is not None:
            f.write(f"Model Performance:\n")
            f.write(f"- Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})\n")
            if test_accuracy is not None:
                f.write(f"- Test set accuracy: {test_accuracy:.4f}\n")
            if f1_score is not None:
                f.write(f"- F1 Score: {f1_score:.4f}\n")
            if roc_auc is not None:
                f.write(f"- ROC AUC: {roc_auc:.4f}\n")
            if class_weights is not None:
                f.write(f"- Class weights: {class_weights}\n")
            f.write("\n")
        
        # Classification report
        if classification_report:
            f.write("Classification Report:\n")
            f.write(f"{classification_report}\n\n")
        
        # Feature importance
        f.write("Top 20 Most Important Features:\n")
        readable_feature_importance = create_readable_feature_importance(feature_importance, tfidf_vectorizer)
        for i, (feature, importance, description) in enumerate(readable_feature_importance[:20], 1):
            f.write(f"{i}. {description} ({importance:.2%})\n")
        f.write("\n")
        
        # Visualization references
        f.write("Generated Visualizations:\n")
        f.write(f"1. Feature Importance Chart: {results_dir}/feature_importance.png\n")
        f.write(f"   - Shows the relative importance of features for predicting project success\n")
        
        if pydotplus_available:
            f.write(f"2. Decision Tree Diagram: {results_dir}/decision_tree.png\n")
            f.write(f"   - Visualizes the decision-making process used to classify projects\n")
            
            # Add detailed explanation of how to read the decision tree
            f.write("\nHow to Read the Decision Tree Diagram:\n")
            f.write("- Start at the top node and follow the tree downward\n")
            f.write("- At each decision node, if the condition is true, follow the left branch; if false, follow the right branch\n")
            f.write("- Colored leaf nodes show the classification outcome (green = success, red = failure)\n")
            f.write("- Darker colors indicate higher confidence in the prediction\n")
            f.write("- Each node shows what percentage of training samples follow that path\n")
            f.write("- The most important factors in the model appear near the top of the tree\n\n")
            
            f.write(f"3. Live Projects Status: {results_dir}/live_projects_status.png\n")
            f.write(f"   - Plots live projects based on their funding status and time remaining\n\n")
        else:
            f.write(f"2. Live Projects Status: {results_dir}/live_projects_status.png\n")
            f.write(f"   - Plots live projects based on their funding status and time remaining\n\n")
        
        # Live project analysis
        if live_projects_analysis:
            f.write("Live Projects Analysis:\n")
            
            # Summarize prediction results
            likely_success = sum(1 for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to succeed')
            likely_fail = sum(1 for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to fail')
            unsure = sum(1 for p in live_projects_analysis if p.get('prediction_outcome') == 'Prediction uncertain')
            
            f.write(f"- {likely_success} projects likely to succeed\n")
            f.write(f"- {likely_fail} projects likely to fail\n")
            f.write(f"- {unsure} projects with uncertain prediction\n\n")
            
            # List projects by prediction status
            f.write("Projects Likely to Succeed:\n")
            for proj in [p for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to succeed']:
                f.write(f"- {proj['title']} ({proj.get('days_left', 'unknown')} days left)\n")
                f.write(f"  Funded: {proj.get('funding_percent', 0):.1%}, "
                     f"Remarkable features: {', '.join(proj.get('remarkable_features', ['None']))}\n")
            
            f.write("\nProjects Likely to Fail:\n")
            for proj in [p for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to fail']:
                f.write(f"- {proj['title']} ({proj.get('days_left', 'unknown')} days left)\n")
                f.write(f"  Funded: {proj.get('funding_percent', 0):.1%}, "
                     f"Remarkable features: {', '.join(proj.get('remarkable_features', ['None']))}\n")
            
            if unsure > 0:
                f.write("\nProjects with Uncertain Prediction:\n")
                for proj in [p for p in live_projects_analysis if p.get('prediction_outcome') == 'Prediction uncertain']:
                    f.write(f"- {proj['title']} ({proj.get('days_left', 'unknown')} days left)\n")
                    f.write(f"  Funded: {proj.get('funding_percent', 0):.1%}, "
                         f"Remarkable features: {', '.join(proj.get('remarkable_features', ['None']))}\n")
        
        f.write("\n=== End of Analysis ===\n")
    
    print(f"Analysis summary saved to {file_path}")
    return file_path

def analyze_live_projects(live_projects, model, feature_names):
    """Analyze live projects using funding/time ratios and model predictions."""
    results = []
    
    # Define thresholds for remarkable features
    thresholds = {
        'num_updates': 5,  # More than 5 updates is remarkable
        'updates_per_day': 0.3,  # More than 0.3 updates per day is remarkable
        'backers_count': 100,  # More than 100 backers is remarkable
        'backers_per_day': 10,  # More than 10 backers per day is remarkable
        'funding_percent': 0.8,  # More than 80% funded is remarkable
        'funding_duration': 30,  # More than 30 days campaign is remarkable
        'total_likes': 50,  # More than 50 likes is remarkable
        'total_comments': 20,  # More than 20 comments is remarkable
        'average_likes_per_update': 10,  # More than 10 likes per update is remarkable
        'average_comments_per_update': 5,  # More than 5 comments per update is remarkable
        'avg_pledge_amount': 50,  # More than $50 average pledge is remarkable
    }
    
    for features, project in live_projects:
        project_copy = project.copy()
        features_copy = features.copy()
        
        # Extract key metrics from project
        days_left = project.get('days_left', 0)
        backers = project.get('backers_count', 0) if 'backers_count' in project else features.get('backers_count', 0)
        pledged = project.get('clean_pledged_amount', 0)
        goal = project.get('clean_funding_goal', 1)  # Default to 1 to avoid division by zero
        
        # Calculate funding percentage
        funding_percent = pledged / goal if goal > 0 else 0
        project_copy['funding_percent'] = funding_percent
        
        # Determine status based on funding percentage
        if funding_percent >= 1.0:
            funding_status = "Funded"
        elif funding_percent >= 0.75:
            funding_status = "Nearly Funded"
        elif funding_percent >= 0.5:
            funding_status = "Halfway Funded"
        elif funding_percent >= 0.25:
            funding_status = "Quarter Funded"
        else:
            funding_status = "Early Funding"
        
        project_copy['funding_status'] = funding_status
        
        # Determine prediction outcome based on funding and time
        if funding_percent >= 1.0:
            prediction_outcome = "Likely to succeed"
            prediction_details = "Already funded!"
        else:
            # Try to calculate time-based projection
            funded_days = project.get('funding_duration', 30) - days_left
            percent_time = funded_days / project.get('funding_duration', 30) if project.get('funding_duration', 30) > 0 else 0.5
            
            if percent_time > 0 and percent_time < 1:
                # Calculate projected final percentage based on current trajectory
                projected_final_percent = funding_percent / percent_time if percent_time > 0 else funding_percent
                
                if projected_final_percent >= 1.0:
                    prediction_outcome = "Likely to succeed"
                    prediction_details = f"On track to be {projected_final_percent:.1%} funded by end date"
                else:
                    prediction_outcome = "Likely to fail"
                    prediction_details = f"Projected to reach only {projected_final_percent:.1%} of goal"
            else:
                # Fallback if we can't calculate time projection
                if funding_percent >= 0.5:
                    prediction_outcome = "Likely to succeed"
                    prediction_details = "Good progress so far"
                else:
                    prediction_outcome = "Likely to fail"
                    prediction_details = "Slow funding progress"
        
        # Override prediction with model if available
        if model is not None:
            # Create feature vector using the features dict we already have
            # Make sure all expected features exist
            features_for_model = {}
            for feature_name in feature_names:
                if feature_name in features:
                    features_for_model[feature_name] = features[feature_name]
                else:
                    features_for_model[feature_name] = 0
            
            # Convert to vector in the correct order
            X = np.array([features_for_model.get(feature, 0) for feature in feature_names]).reshape(1, -1)
            
            # Add dummy features for text fields if expected
            expected_features = len(feature_names)
            if X.shape[1] < expected_features:
                padding = np.zeros((1, expected_features - X.shape[1]))
                X = np.hstack([X, padding])
            
            # Make prediction
            try:
                prediction = model.predict_proba(X)[0]
                confidence = max(prediction)
                
                # Only override if confidence is high enough
                if confidence >= 0.6:
                    predicted_class = model.predict(X)[0]
                    if predicted_class == 1:
                        prediction_outcome = "Likely to succeed"
                        prediction_details = f"Model predicts success with {confidence:.1%} confidence"
                    else:
                        prediction_outcome = "Likely to fail"
                        prediction_details = f"Model predicts failure with {confidence:.1%} confidence"
                else:
                    prediction_outcome = "Prediction uncertain"
                    prediction_details = f"Model confidence too low ({confidence:.1%})"
            except Exception as e:
                print(f"Error predicting for project {project.get('title', 'unknown')}: {e}")
                # Keep the funding/time based prediction
        
        project_copy['prediction_outcome'] = prediction_outcome
        project_copy['prediction_details'] = prediction_details
        
        # Identify remarkable features for this project
        remarkable_features = []
        
        # Check each metric against thresholds
        for metric, threshold in thresholds.items():
            metric_value = None
            
            # Look for the metric in both project and features dictionaries
            if metric in project and isinstance(project[metric], (int, float)):
                metric_value = project[metric]
            elif metric in features and isinstance(features[metric], (int, float)):
                metric_value = features[metric]
            elif metric == 'funding_percent':
                metric_value = funding_percent
            
            if metric_value is not None and metric_value >= threshold:
                readable_metric = {
                    'num_updates': 'high update count',
                    'updates_per_day': 'frequent updates',
                    'backers_count': 'many backers',
                    'backers_per_day': 'fast backer growth',
                    'funding_percent': 'well funded',
                    'funding_duration': 'long campaign',
                    'total_likes': 'highly liked',
                    'total_comments': 'heavily commented',
                    'average_likes_per_update': 'engaging updates',
                    'average_comments_per_update': 'interactive community',
                    'avg_pledge_amount': 'high average pledge'
                }.get(metric, metric)
                
                remarkable_features.append(readable_metric)
        
        # Add engagement score based on likes and comments
        engagement_score = (
            features.get('total_likes', 0) * 0.5 + 
            features.get('total_comments', 0) * 2 + 
            features.get('average_likes_per_update', 0) * 5
        ) / 100.0
        
        if engagement_score >= 1.0:
            remarkable_features.append('exceptional engagement')
        elif engagement_score >= 0.5:
            remarkable_features.append('good engagement')
        
        project_copy['remarkable_features'] = remarkable_features
        project_copy['engagement_score'] = engagement_score
        
        results.append(project_copy)
    
    # Sort by likelihood of success
    results.sort(key=lambda x: (
        0 if x['prediction_outcome'] == 'Likely to succeed' else 
        1 if x['prediction_outcome'] == 'Prediction uncertain' else 2,
        -x.get('funding_percent', 0)
    ))
    
    return results

def visualize_feature_importance(importance_df, top_n=10, results_dir=None, tfidf_vectorizer=None):
    """Create a horizontal bar chart of feature importance."""
    plt.figure(figsize=(12, 8))
    
    # Get top N features
    top_features = importance_df.head(top_n).copy()
    
    # Create readable labels for features
    readable_labels = {}
    
    # Base feature labels
    for feature, label in FEATURE_LABELS.items():
        readable_labels[feature] = label
    
    # Add TF-IDF feature labels (actual words)
    if tfidf_vectorizer is not None:
        # Create reverse mapping from index to word
        idx_to_word = {idx: word for word, idx in tfidf_vectorizer.vocabulary_.items()}
        
        # Map TF-IDF features to their actual words
        for feature in top_features['feature']:
            if feature.startswith('tfidf_'):
                try:
                    idx = int(feature.split('_')[1])
                    word = idx_to_word.get(idx)
                    if word:
                        readable_labels[feature] = f"Word '{word}'"
                except (ValueError, IndexError):
                    pass
    
    # Apply readable labels to features
    top_features['readable_name'] = top_features['feature'].map(
        lambda x: readable_labels.get(x, x)
    )
    
    # Sort by importance for the plot
    top_features = top_features.sort_values('importance', ascending=True)
    
    # Plot
    ax = sns.barplot(x='importance', y='readable_name', data=top_features, 
                    palette=[COLORS['primary']] * len(top_features))
    
    # Add percentage to the bars
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        plt.text(width + 0.01, p.get_y() + p.get_height()/2, 
                f'{width:.1%}', ha='left', va='center')
    
    # Styling
    plt.title('Top Factors Influencing Project Success', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('')
    plt.tight_layout()
    
    # Save if results directory is provided
    if results_dir:
        plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        
    return plt.gcf()

def visualize_live_projects_status(live_projects_analysis, results_dir=None):
    """Create a scatter plot showing the status of live projects."""
    # Set figure size - making it taller to accommodate the text at bottom
    plt.figure(figsize=(16, 16), facecolor='white')  # Increased height to make room for legend at bottom
    
    # Create subplot with extra space at bottom for legend and explanation text
    ax = plt.subplot2grid((8, 1), (0, 0), rowspan=6)  # Changed from 7,1 to 8,1 for more space
    
    # Extract data
    funded_percent = [p.get('percent_funded', 0) for p in live_projects_analysis]
    time_percent = [p.get('percent_time', 0) for p in live_projects_analysis]
    outcomes = [p.get('prediction_outcome', 'Unknown') for p in live_projects_analysis]
    
    # Set maximum y-axis limit for better visualization
    max_y_display = 300  # Fixed cap at 300% for readability
    
    # Add jitter to improve visibility of overlapping points
    jitter_amount = 1.5
    jittered_time = [t + (np.random.random() - 0.5) * jitter_amount for t in time_percent]
    
    # Set style for cleaner plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.grid(True, alpha=0.25, linestyle='-')
    
    # Use distinct colors with higher contrast
    outcome_colors = {
        'Likely to succeed': '#17A589',  # Darker green
        'Likely to fail': '#C0392B',     # Darker red
        'Prediction uncertain': '#7F8C8D' # Darker gray
    }
    
    # Create scatter plot with larger markers
    for outcome in ['Likely to succeed', 'Likely to fail', 'Prediction uncertain']:
        indices = [i for i, o in enumerate(outcomes) if o == outcome]
        if not indices:
            continue
            
        plt.scatter(
            [jittered_time[i] for i in indices],
            [min(funded_percent[i], max_y_display) for i in indices],
            c=outcome_colors.get(outcome, '#7F8C8D'),
            label=outcome,
            alpha=0.85,
            s=150,  # Larger point size
            edgecolors='white',
            linewidths=1.0,
            zorder=3
        )
    
    # Add reference line (y=x) with DASHED styling (clearly labeled)
    plt.plot([0, 100], [0, 100], 
             linestyle='--', 
             color='#34495E', 
             alpha=0.7, 
             linewidth=2.5,
             label='On-track Reference Line (Diagonal)',
             zorder=2)
    
    # Mark 100% funding threshold with DOTTED horizontal line (clearly labeled)
    plt.axhline(
        y=100, 
        color='#34495E', 
        linestyle=':', 
        alpha=0.7,
        linewidth=2.5,
        label='100% Funding Goal (Horizontal)',
        zorder=2
    )
    
    # Add 90% trajectory reference line (where projects become "likely to succeed")
    plt.plot([0, 100], [0, 90], 
             linestyle='-.',  # Dot-dash line 
             color='#2E86C1',
             alpha=0.5, 
             linewidth=1.5,
             label='90% Trajectory Threshold',
             zorder=2)
    
    # Add clear regions
    plt.axvspan(0, 100, ymin=0, ymax=100/max_y_display, 
                alpha=0.1, color='#C0392B', zorder=1)  # Below line: risky
    plt.axvspan(0, 100, ymin=100/max_y_display, ymax=1, 
                alpha=0.1, color='#17A589', zorder=1)  # Above line: on track
    
    # Styling
    plt.title('Live Projects: Funding Progress vs. Time Elapsed', fontsize=22, fontweight='bold')
    plt.xlabel('Percentage of Campaign Time Elapsed', fontsize=16, fontweight='bold')
    plt.ylabel('Percentage of Funding Goal Achieved', fontsize=16, fontweight='bold')
    
    # Set axis limits and ticks
    plt.xlim(0, 100)
    plt.ylim(0, max_y_display)
    plt.xticks(np.arange(0, 101, 10), fontsize=12)
    plt.yticks(np.arange(0, max_y_display+1, 50), fontsize=12)
    
    # Get handles and labels before creating or removing any legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Remove the legend from the main plot if it exists
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    
    # Create a legend subplot below the main chart
    legend_ax = plt.subplot2grid((8, 1), (6, 0), rowspan=1)
    legend_ax.axis('off')  # Hide the axes
    
    # Move the legend to below the chart with improved visibility
    legend = legend_ax.legend(
        handles, 
        labels,
        title='Project Status & Reference Lines',
        loc='center',  # Center in the legend subplot
        fontsize=14,
        ncol=3,  # Display in 3 columns for better horizontal layout
        framealpha=0.95,  # Higher opacity for background
        edgecolor='#34495E',  # Darker edge color
        facecolor='#F8F9FA',  # Light gray background color
        frameon=True,  # Ensure frame is visible
        borderpad=1,  # More padding inside the legend
        fancybox=True,
        shadow=True,
        labelspacing=1.2  # More space between legend items
    )
    legend.get_title().set_fontsize(16)
    legend.get_title().set_fontweight('bold')
    legend.get_frame().set_linewidth(2)  # Thicker border
    
    # Add clear explanatory text in a separate subplot below the main chart
    # Create a new subplot for the explanation text
    explanation_ax = plt.subplot2grid((8, 1), (7, 0), rowspan=1)
    explanation_ax.axis('off')  # Hide the axes
    
    # Add a box with the explanation text
    explanation_text = (
        "Reference Lines Explained:\n"
        "• Diagonal Dashed Line: Projects raising funds at the ideal rate (funding % = time %)\n"
        "• Horizontal Dotted Line: 100% funding goal achievement\n"
        "• Projects above diagonal line are on track to succeed\n"
        "• Projects below diagonal line need to increase their funding rate"
    )
    
    explanation_box = plt.text(
        0.5, 0.5, 
        explanation_text,
        fontsize=14,
        ha='center',
        va='center',
        bbox=dict(
            facecolor='#F8F9FA', 
            alpha=0.95,
            edgecolor='#34495E',
            boxstyle='round,pad=0.8',
            linewidth=2
        ),
        transform=explanation_ax.transAxes
    )
    
    plt.tight_layout()
    
    # Save if results directory is provided
    if results_dir:
        plt.savefig(os.path.join(results_dir, 'live_projects_status.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        
    return plt.gcf()

def create_decision_tree_diagram(X, y, feature_names, class_names=['Fail', 'Succeed'], max_depth=3, results_dir=None):
    """Create and visualize a decision tree to show the decision flow."""
    # Check if pydotplus is available
    if not pydotplus_available:
        print("Decision tree visualization skipped: pydotplus not available")
        return None
    
    # Train a separate decision tree model on the top features for visualization
    # This avoids the issue of feature count mismatch
    try:
        # Create a new decision tree using only the selected features
        # Limit max_depth to 3 for better readability
        dtree = DecisionTreeClassifier(
            max_depth=max_depth, 
            class_weight='balanced', 
            random_state=42,
            min_samples_split=5  # Increase to reduce complexity
        )
        
        # Get indices of the top features if we're using a subset
        if len(feature_names) < X.shape[1]:
            print(f"Creating decision tree with a subset of {len(feature_names)} features (from {X.shape[1]} total)")
            # Get the column indices that correspond to the selected features
            X_subset = X[:, :len(feature_names)]  # Just use the first N features
            # Train on the subset
            dtree.fit(X_subset, y)
        else:
            # Use all features
            dtree.fit(X, y)
        
        # Create more readable feature names for the visualization
        readable_feature_names = []
        for feature in feature_names:
            if feature in FEATURE_LABELS:
                readable_feature_names.append(FEATURE_LABELS[feature])
            elif feature.startswith('tfidf_'):
                readable_feature_names.append(f"Word Feature {feature.split('_')[1]}")
            else:
                readable_feature_names.append(feature.replace('_', ' ').title())
        
        # Export tree as DOT file with improved styling
        dot_data = StringIO()
        export_graphviz(
            dtree, 
            out_file=dot_data, 
            feature_names=readable_feature_names,
            class_names=class_names,
            filled=True, 
            rounded=True,
            special_characters=True,
            proportion=True,
            impurity=False,  # Hide impurity to reduce clutter
            precision=1      # Fewer decimal places for cleaner look
        )
        
        # Convert to graph and render as image
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        
        # Improve graph styling
        graph.set_rankdir('TB')  # Top to bottom layout (more readable than left to right)
        graph.set_fontname('Arial')
        graph.set_fontsize('14')
        
        # Enhance node appearance with simpler styling (avoid set_fontweight)
        for i, node in enumerate(graph.get_nodes()):
            if node.get_name() not in ('node', 'edge'):
                # Set default styling for all nodes
                node.set_fontname('Arial')
                node.set_fontsize('12')
                
                # Extract node text for specific styling
                node_text = node.get_attributes().get('label', '')
                
                # Style leaf nodes (class predictions) differently
                if 'class' in node_text:
                    # Make leaf nodes larger
                    node.set_fontsize('14')
                    node.set_penwidth('2.0')
                    
                    if 'value = [' in node_text:
                        # Extract class distribution values
                        values_str = node_text.split('value = [')[1].split(']')[0]
                        values = [float(x) for x in values_str.split(',')]
                        
                        # Calculate color based on class dominance with stronger contrast
                        if len(values) == 2:
                            fail_ratio, succeed_ratio = values[0] / sum(values), values[1] / sum(values)
                            
                            # Set color with stronger contrast based on confidence
                            if succeed_ratio > 0.75:  # Strong success prediction
                                color = '#1a9850'  # Darker green for high confidence
                            elif succeed_ratio > 0.5:  # Moderate success prediction
                                color = '#91cf60'  # Lighter green for lower confidence
                            elif succeed_ratio >= 0.25:  # Uncertain, but leaning failure
                                color = '#fc8d59'  # Lighter red/orange for lower confidence
                            else:  # Strong failure prediction
                                color = '#d73027'  # Darker red for high confidence
                                
                            node.set_fillcolor(color)
        
        # Set graph size for better visibility
        graph.set_size('"12,8"')
        graph.set_dpi('300')
        
        # Save if results directory is provided
        if results_dir:
            output_path = os.path.join(results_dir, 'decision_tree.png')
            graph.write_png(output_path)
            print(f"Enhanced decision tree visualization saved to {output_path}")
        
        return graph
    except Exception as e:
        print(f"Error creating decision tree diagram: {e}")
        return None

def create_decision_tree_visualization(X_train, y_train, importance_df, results_dir):
    """Create and save a decision tree visualization."""
    print("\nCreating decision tree diagram...")
    top_features = importance_df.nlargest(10, 'importance')['feature'].tolist()
    
    # Create the technical decision tree only
    tree_result = create_decision_tree_diagram(
        X_train, y_train, 
        feature_names=top_features,
        results_dir=results_dir
    )
    
    if tree_result is None:
        print("Note: Decision tree visualization could not be created.")

def main():
    """Run the complete crowdfunding analysis pipeline."""
    # Create results directory
    results_dir = create_results_dir()
    
    # Load project data
    projects = load_and_validate_data()
    if not projects:
        return
    
    # Calculate basic project statistics
    project_stats = {
        'total': len(projects),
        'live': len(projects),
        'successful': sum(1 for p in projects if p['success'] == 1),
        'failed': sum(1 for p in projects if p['success'] == 0)
    }
    print_project_statistics(project_stats)
    
    # Create dataset and select features
    features_list, labels, live_projects = create_dataset(projects, for_training=True)
    baseline_features = select_features(features_list)
    
    # Create feature matrix and train model
    X, y, feature_names, tfidf = prepare_feature_matrix(features_list, baseline_features, labels)
    
    # Skip if not enough data
    if len(set(y)) < 2 or len(y) < 10:
        print("Not enough diverse data for model training. Skipping model training.")
        analyze_without_model(feature_names, live_projects, results_dir, project_stats)
        return
    
    # Train and evaluate model
    model, metrics = train_and_evaluate_model(X, y, feature_names)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(model, feature_names, results_dir, tfidf)
    
    # Visualize decision tree
    create_decision_tree_visualization(X, y, importance_df, results_dir)
    
    # Analyze live projects
    live_analysis = analyze_live_projects(live_projects, model, feature_names)
    visualize_live_projects_status(live_analysis, results_dir)
    
    # Save results
    save_complete_analysis(
        results_dir, importance_df, metrics['classification_report'], 
        tfidf, live_analysis, metrics, project_stats
    )
    
    # Print key findings
    print_key_findings(importance_df, metrics, live_analysis, results_dir, tfidf)

def load_and_validate_data():
    """Load project data from the appropriate directory."""
    # Define paths - using best practices to find data directory
    data_dir = 'crowdfunding-analysis/webscraper/scrapers/scraped_data'
    
    # If that doesn't work, try to build an absolute path based on script location
    if not os.path.exists(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        alternate_paths = [
            os.path.join(parent_dir, "webscraper", "scrapers", "scraped_data"),
            os.path.join(os.path.dirname(script_dir), "webscraper", "scrapers", "scraped_data")
        ]
        
        for path in alternate_paths:
            if os.path.exists(path):
                data_dir = path
                break
    
    # Load data
    print("Loading project data...")
    projects = load_project_data(data_dir)
    
    return projects

def print_project_statistics(stats):
    """Print statistics about the loaded projects."""
    print(f"Loaded {stats['total']} projects: {stats['live']} live")
    print(f"Success breakdown: {stats['successful']} successful/predicted to succeed, {stats['failed']} failed/predicted to fail")

def select_features(features_list):
    """Select appropriate features for model training."""
    # Define baseline features that all projects have
    # REMOVED highly predictive features: backers_count, backers_per_day
    baseline_features = [
        'num_updates', 'total_likes', 'total_comments', 
        'total_comment_count', 'funding_duration', 'updates_per_day',
        'avg_update_length', 'days_left',
        'has_first_day_update', 'has_first_week_update',
        'average_likes_per_update', 'average_comments_per_update',
        'avg_pledge_amount'  # Kept avg_pledge_amount but removed backers_per_day
    ]
    
    # Add project-specific features
    excluded_features = {
        'all_updates_text', 'percent_funded', 'projected_final_percent', 
        'backers_count', 'backers_per_day'
    }
    
    for project in features_list:
        # Add features present in the dataset not already in baseline_features
        for key in project.keys():
            if (key not in baseline_features and 
                key not in excluded_features):
                baseline_features.append(key)
    
    print(f"Using {len(baseline_features)} features after careful feature selection")
    print(f"Excluded direct predictors: backers_count, backers_per_day")
    print(f"Included more nuanced indicator: avg_pledge_amount")
    
    return baseline_features

def prepare_feature_matrix(features_list, baseline_features, labels):
    """Prepare feature matrix for model training."""
    # Convert features to DataFrame
    df = pd.DataFrame(features_list)
    
    # Ensure all necessary columns exist
    for feature in baseline_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Create TF-IDF features from update text
    print("Creating text features...")
    tfidf = TfidfVectorizer(max_features=100, min_df=2)
    
    # Handle empty text fields
    text_data = df['all_updates_text'].fillna('').tolist()
    if all(not text for text in text_data):
        print("Warning: All update texts are empty. Using only numerical features.")
        text_features = np.zeros((len(text_data), 1))  # Dummy feature
    else:
        text_features = tfidf.fit_transform(text_data)
    
    # Combine numerical and text features
    numerical_features = df[baseline_features].fillna(0).values
    X = np.hstack([numerical_features, text_features.toarray()])
    y = np.array(labels)
    
    # Get feature names
    feature_names = baseline_features + [f'tfidf_{i}' for i in range(text_features.shape[1])]
    
    return X, y, feature_names, tfidf

def train_and_evaluate_model(X, y, feature_names):
    """Train and evaluate the Random Forest model."""
    # Handle class imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y), y=y
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"Class weights to address imbalance: {class_weight_dict}")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train the model
    print("Training Random Forest model with cross-validation...")
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Standard number of trees
        max_features='sqrt',  # Standard scikit-learn default
        min_samples_split=2,  # Default value
        min_samples_leaf=1,  # Default value - allows full growth
        max_depth=None,  # Unlimited depth
        bootstrap=True,  # Standard approach
        random_state=42,
        class_weight=class_weight_dict  # Keep class balancing
    )
    
    # Perform stratified k-fold cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Train on full training set for final evaluation
    rf_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'classification_report': classification_report(y_test, y_pred),
        'cv_scores': cv_scores
    }
    
    # Add ROC AUC if possible
    if len(set(y_test)) > 1:  # Only calculate ROC AUC if there are both classes
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Print evaluation metrics
    print("\nModel Evaluation:")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    return rf_model, metrics
    
def analyze_feature_importance(model, feature_names, results_dir, tfidf_vectorizer=None):
    """Analyze and visualize feature importance."""
    # Get feature importance from model
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Display top features
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))
    
    # Create and save visualization
    visualize_feature_importance(importance, results_dir=results_dir, tfidf_vectorizer=tfidf_vectorizer)
    
    return importance

def analyze_without_model(feature_names, live_projects, results_dir, project_stats):
    """Analyze projects when there's not enough data for model training."""
    # Just analyze live projects based on time/funding ratio
    live_projects_analysis = analyze_live_projects(live_projects, None, feature_names)
    
    save_analysis_summary(
        results_dir, 
        pd.DataFrame({
            'feature': feature_names,
            'importance': [1] * len(feature_names)
        }), 
        "No model trained", 
        None, 
        live_projects_analysis,
        projects_summary=project_stats
    )
    
    # Create visualization of live projects status
    visualize_live_projects_status(live_projects_analysis, results_dir)
    
    print(f"\nResults have been saved to: {results_dir}")
    
def save_complete_analysis(results_dir, importance_df, classification_report, 
                           tfidf, live_analysis, metrics, project_stats):
    """Save complete analysis results."""
    save_analysis_summary(
        results_dir, 
        importance_df, 
        classification_report, 
        tfidf, 
        live_analysis,
        cv_scores=metrics.get('cv_scores'), 
        test_accuracy=metrics.get('accuracy'), 
        f1_score=metrics.get('f1_score'), 
        roc_auc=metrics.get('roc_auc'),
        class_weights=metrics.get('class_weight_dict'),
        projects_summary=project_stats
    )
    
    print(f"\nResults have been saved to: {results_dir}")

def print_key_findings(importance_df, metrics, live_projects_analysis, results_dir, tfidf):
    """Print a summary of key findings."""
    print("\nKey findings:")
    
    # Top features
    top_features = importance_df.head(3)['feature'].tolist()
    readable_features = []
    
    for feature in top_features:
        if feature in FEATURE_LABELS:
            readable_features.append(FEATURE_LABELS[feature])
        elif feature.startswith('tfidf_'):
            # For TF-IDF features, try to get the actual word from the vectorizer
            try:
                idx = int(feature.split('_')[1])
                word = [word for word, i in tfidf.vocabulary_.items() if i == idx]
                if word:
                    readable_features.append(f"Word '{word[0]}'")
                else:
                    readable_features.append(feature)
            except:
                readable_features.append(feature)
        else:
            readable_features.append(feature)
    
    top_features_formatted = ', '.join(readable_features)
    
    print(f"1. Top factors influencing project success: {top_features_formatted}")
    print(f"2. Cross-validation accuracy: {metrics['cv_scores'].mean():.1%} (more reliable than test accuracy)")
    
    # Live project predictions
    likely_success = sum(1 for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to succeed')
    likely_fail = sum(1 for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to fail')
    print(f"3. Live projects: {likely_success} likely to succeed, {likely_fail} likely to fail")
    
    # Output locations
    print("\nCheck the results directory for detailed analysis and visualizations in:")
    print(f"- {results_dir}/analysis_summary.txt (Text report)")
    print(f"- {results_dir}/feature_importance.png (Feature importance visualization)")
    if pydotplus_available:
        print(f"- {results_dir}/decision_tree.png (Decision flow diagram)")
    print(f"- {results_dir}/live_projects_status.png (Projects funding status visualization)")

if __name__ == "__main__":
    main() 