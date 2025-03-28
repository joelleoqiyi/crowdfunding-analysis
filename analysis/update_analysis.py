import json
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import re
from datetime import datetime, timezone, timedelta
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Try to import pydotplus but make it optional
pydotplus_available = False
try:
    import pydotplus
    import matplotlib.colors as mcolors
    pydotplus_available = True
except ImportError:
    print("pydotplus not available. Decision tree visualization will be disabled.")
    # Create a dummy mcolors if not available
    class DummyColors:
        def __getattr__(self, name):
            return '#000000'  # Default color
    mcolors = DummyColors()

# Define color scheme for visualizations
COLORS = {
    'primary': '#1f77b4',    # Main color for plots
    'success': '#2ca02c',    # Color for successful projects
    'failure': '#d62728',    # Color for failed projects
    'neutral': '#7f7f7f',    # Neutral color
    'highlight': '#ff7f0e',  # Highlighting important elements
    'background': '#f5f5f5'  # Background color
}

def load_project_data(data_dir):
    """Load live project data from a single directory."""
    projects = []
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found")
        # Try to find the correct path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        potential_path = os.path.join(parent_dir, "webscraper", "scrapers", "scraped_data")
        if os.path.exists(potential_path):
            print(f"Found data directory at: {potential_path}")
            data_dir = potential_path
        else:
            print("Could not locate scraped_data directory. Please check the path.")
            return []
    
    print(f"Loading data from: {data_dir}")
    # Load all live projects from the directory
    file_count = 0
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_count += 1
            try:
                with open(os.path.join(data_dir, filename), 'r') as f:
                    data = json.load(f)
                    
                    # Skip if no campaign details
                    if 'campaign_details' not in data or not data['campaign_details']:
                        print(f"Skipping {filename} - no campaign details")
                        continue
                    
                    campaign = data['campaign_details']
                    
                    # Only process live projects (those with days_left field)
                    if 'days_left' not in campaign:
                        continue
                    
                    # Add title for better project identification in reports
                    if 'url' in data:
                        # Extract project name from URL
                        project_name = data['url'].split('/')[-1]
                        data['title'] = project_name
                    
                    # Clean monetary values
                    funding_goal = campaign.get('funding_goal', '0')
                    pledged_amount = campaign.get('pledged_amount', '0')
                    
                    # Clean strings and convert to float
                    if isinstance(funding_goal, str):
                        funding_goal = float(re.sub(r'[^\d.]', '', funding_goal.replace(',', '')) or 0)
                    if isinstance(pledged_amount, str):
                        pledged_amount = float(re.sub(r'[^\d.]', '', pledged_amount.replace(',', '')) or 0)
                    
                    # Store cleaned values
                    data['clean_funding_goal'] = funding_goal
                    data['clean_pledged_amount'] = pledged_amount
                    data['is_live'] = True

                    # Calculate percentage of funding achieved
                    percent_funded = (pledged_amount / funding_goal * 100) if funding_goal > 0 else 0
                    data['percent_funded'] = percent_funded
                    
                    # Calculate percent of time elapsed
                    try:
                        if 'funding_start_date' in campaign and 'funding_end_date' in campaign:
                            start_date = datetime.fromisoformat(campaign['funding_start_date'].replace('Z', '+00:00'))
                            end_date = datetime.fromisoformat(campaign['funding_end_date'].replace('Z', '+00:00'))
                            total_duration = (end_date - start_date).total_seconds()
                            
                            # Use current date or calculate from days_left
                            current_date = datetime.now(timezone.utc)
                            if 'days_left' in campaign:
                                days_left = int(campaign['days_left'])
                                current_date = end_date - timedelta(days=days_left)
                            
                            elapsed = (current_date - start_date).total_seconds()
                            percent_time = (elapsed / total_duration * 100) if total_duration > 0 else 0
                            data['percent_time'] = percent_time
                            
                            # Project final funding
                            if percent_time > 0:
                                projected_final = (percent_funded / percent_time) * 100
                                data['projected_final_percent'] = projected_final
                                
                                # Predict success based on projection
                                if projected_final >= 100:
                                    data['success'] = 1
                                    data['success_prediction'] = f"{percent_funded:.1f}% funded at {percent_time:.1f}% of time - On track to succeed"
                                else:
                                    data['success'] = 0
                                    data['success_prediction'] = f"{percent_funded:.1f}% funded at {percent_time:.1f}% of time - Projected to reach {projected_final:.1f}%"
                            else:
                                data['success'] = 0 if percent_funded < 10 else 1  # Default for new campaigns
                        else:
                            # Can't determine time percentage, use simpler metric
                            data['success'] = 1 if percent_funded >= 50 else 0
                            data['success_prediction'] = f"{percent_funded:.1f}% funded - insufficient date data"
                    except Exception as e:
                        print(f"Error calculating time metrics for {filename}: {str(e)}")
                        data['success'] = 1 if percent_funded >= 50 else 0
                
                    projects.append(data)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {filename}, skipping file")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}, skipping file")
    
    print(f"Found {file_count} JSON files, loaded {len(projects)} live projects successfully")
    return projects

def extract_features(project, for_training=True):
    """Extract relevant features from live project updates."""
    features = {}
    
    # Basic update statistics
    updates = project.get('updates', {})
    features['num_updates'] = updates.get('count', 0)
    
    # Update content analysis
    update_contents = []
    total_likes = 0
    total_comments = 0
    total_comment_count = 0
    update_length_avg = 0
    
    # Extract and process all updates
    for update in updates.get('content', []):
        update_content = update.get('content', '')
        update_contents.append(update_content)
        
        # Count metrics
        total_likes += update.get('likes_count', 0)
        total_comments += len(update.get('comments', []))
        total_comment_count += update.get('comments_count', 0)
        
        # Update length
        if update_content:
            update_length_avg += len(update_content)
    
    # Calculate averages
    num_updates = features['num_updates']
    if num_updates > 0:
        update_length_avg = update_length_avg / num_updates
    
    features['total_likes'] = total_likes
    features['total_comments'] = total_comments
    features['total_comment_count'] = total_comment_count
    features['avg_update_length'] = update_length_avg
    
    # Campaign details metrics
    campaign = project.get('campaign_details', {})
    
    # Get duration or calculate it
    if 'funding_duration_days' in campaign:
        funding_duration = campaign.get('funding_duration_days', 0)
    elif 'funding_start_date' in campaign and 'funding_end_date' in campaign:
        try:
            start_date = datetime.fromisoformat(campaign['funding_start_date'].replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(campaign['funding_end_date'].replace('Z', '+00:00'))
            funding_duration = (end_date - start_date).days
        except:
            funding_duration = 30  # Default assumption
    else:
        funding_duration = 30  # Default assumption
    
    features['funding_duration'] = funding_duration
    
    # Calculate update frequency
    if funding_duration > 0:
        features['updates_per_day'] = features['num_updates'] / funding_duration
    else:
        features['updates_per_day'] = 0
    
    # Add backers count feature
    if 'backers_count' in campaign:
        try:
            backers = campaign['backers_count']
            if isinstance(backers, str):
                backers = float(re.sub(r'[^\d.]', '', backers))
            features['backers_count'] = backers
        except:
            features['backers_count'] = 0
    else:
        features['backers_count'] = 0
        
    # Create early update features
    features['has_first_day_update'] = 0
    features['has_first_week_update'] = 0
    
    # Extract engagement metrics
    features['average_likes_per_update'] = total_likes / max(num_updates, 1)
    features['average_comments_per_update'] = total_comments / max(num_updates, 1)
    
    # Live project metrics - carefully avoiding direct indicators of success
    clean_funding_goal = project.get('clean_funding_goal', 0)
    clean_pledged_amount = project.get('clean_pledged_amount', 0)
    
    # Features that are indirectly related to success, not direct predictors
    if for_training:
        # For training models, we avoid including direct success indicators
        features['backers_per_day'] = features['backers_count'] / max(funding_duration * (project.get('percent_time', 100)/100), 1)
        
        # Average pledge amount doesn't directly indicate success
        if features['backers_count'] > 0:
            features['avg_pledge_amount'] = clean_pledged_amount / features['backers_count']
        else:
            features['avg_pledge_amount'] = 0

        # Include time metrics but not the direct ratio that determines success
        features['days_left'] = int(campaign.get('days_left', 0))
        features['percent_time'] = project.get('percent_time', 0)
    
    # Combine all update content for text analysis
    features['all_updates_text'] = ' '.join(update_contents)
    
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
    """Create a results directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'crowdfunding-analysis/analysis/results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def create_readable_feature_importance(importance_df, tfidf_vectorizer=None):
    """Create a more readable version of feature importance rankings.
    
    Returns a list of tuples (feature_name, importance_value, description)
    """
    # Sort by importance
    if 'importance' in importance_df.columns:
        importance_df = importance_df.sort_values('importance', ascending=False).copy()
    
    result = []
    
    # Create descriptive labels
    feature_labels = {
        'funding_duration': 'Campaign Duration',
        'total_likes': 'Total Likes',
        'total_comments': 'Total Comments',
        'total_comment_count': 'Comment Count',
        'num_updates': 'Number of Updates',
        'updates_per_day': 'Updates per Day',
        'avg_update_length': 'Average Update Length',
        'backers_count': 'Number of Backers',
        'backers_per_day': 'Backers per Day',
        'avg_pledge_amount': 'Average Pledge Amount',
        'average_likes_per_update': 'Average Likes per Update',
        'average_comments_per_update': 'Average Comments per Update',
        'days_left': 'Days Left in Campaign',
        'has_first_day_update': 'Has Update on First Day',
        'has_first_week_update': 'Has Update in First Week'
    }
    
    # Process each feature
    for _, row in importance_df.iterrows():
        feature = row['feature']
        importance = row['importance'] if 'importance' in row else 0
        
        # Get description based on feature type
        if feature in feature_labels:
            description = feature_labels[feature]
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

def visualize_feature_importance(importance_df, top_n=10, results_dir=None):
    """Create a horizontal bar chart of feature importance."""
    plt.figure(figsize=(12, 8))
    
    # Get top N features
    top_features = importance_df.head(top_n).copy()
    
    # Create readable labels
    feature_labels = {
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
        'days_left': 'Days Left in Campaign'
    }
    
    # Create readable labels for features
    top_features['readable_name'] = top_features['feature'].map(
        lambda x: feature_labels.get(x, x)
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
    plt.figure(figsize=(14, 10))
    
    # Extract data
    funded_percent = [p.get('percent_funded', 0) for p in live_projects_analysis]
    time_percent = [p.get('percent_time', 0) for p in live_projects_analysis]
    projected_final = [p.get('projected_final_percent', 0) for p in live_projects_analysis]
    outcomes = [p.get('prediction_outcome', 'Unknown') for p in live_projects_analysis]
    
    # Create color map
    colors = [COLORS['success'] if o == 'Likely to succeed' else 
              COLORS['failure'] if o == 'Likely to fail' else 
              COLORS['neutral'] for o in outcomes]
    
    # Plot data
    plt.scatter(time_percent, funded_percent, c=colors, alpha=0.7, s=100)
    
    # Add reference line (y=x)
    max_val = max(max(funded_percent, default=100), max(time_percent, default=100))
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    # Add annotations for interesting projects
    for i, (f, t, p, o) in enumerate(zip(funded_percent, time_percent, projected_final, outcomes)):
        if ((f > 150 and t < 50) or  # Far ahead projects
            (f < 10 and t > 50) or   # Far behind projects
            (f > 90 and f < 110 and t > 40 and t < 60)):  # Close to the line
            plt.annotate(f"{f:.0f}% funded\n{p:.0f}% projected", 
                        (t, f), 
                        textcoords="offset points",
                        xytext=(5, 5), 
                        ha='left')
    
    # Styling
    plt.title('Live Projects: Funding Progress vs. Time Elapsed', fontsize=16)
    plt.xlabel('Percentage of Campaign Time Elapsed', fontsize=14)
    plt.ylabel('Percentage of Funding Goal Achieved', fontsize=14)
    plt.xlim(0, 100)
    plt.ylim(0, max(200, max(funded_percent, default=200)))
    plt.grid(alpha=0.3)
    
    # Add legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['success'], markersize=10),
              plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['failure'], markersize=10),
              plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['neutral'], markersize=10),
              plt.Line2D([0], [0], linestyle='--', color='k', alpha=0.5)]
    labels = ['Likely to Succeed', 'Likely to Fail', 'Unknown', 'On-track Reference Line']
    plt.legend(handles, labels, loc='upper left', fontsize=12)
    
    # Add explanatory text
    plt.figtext(0.02, 0.02, 
               "Projects above the dashed line are raising funds faster than needed to succeed.\n"
               "Projects below the line are raising funds slower than needed to succeed.", 
               fontsize=12)
    
    plt.tight_layout()
    
    # Save if results directory is provided
    if results_dir:
        plt.savefig(os.path.join(results_dir, 'live_projects_status.png'), dpi=300, bbox_inches='tight')
        
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
        dtree = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced', random_state=42)
        
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
        
        # Export tree as DOT file
        dot_data = StringIO()
        export_graphviz(
            dtree, 
            out_file=dot_data, 
            feature_names=feature_names,
            class_names=class_names,
            filled=True, 
            rounded=True,
            special_characters=True,
            proportion=True
        )
        
        # Convert to graph and render as image
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        
        # Set colors for nodes
        for i, node in enumerate(graph.get_nodes()):
            if node.get_name() not in ('node', 'edge'):
                node_text = node.get_attributes().get('label', '')
                if 'class' in node_text:
                    if 'value = [' in node_text:
                        # Extract class distribution values
                        values_str = node_text.split('value = [')[1].split(']')[0]
                        values = [float(x) for x in values_str.split(',')]
                        # Calculate color based on class dominance
                        if len(values) == 2:
                            fail_ratio, succeed_ratio = values[0] / sum(values), values[1] / sum(values)
                            if succeed_ratio > 0.5:  # Success dominates
                                color = COLORS['success']
                            else:  # Failure dominates
                                color = COLORS['failure']
                            node.set_fillcolor(color)
        
        # Save if results directory is provided
        if results_dir:
            graph.write_png(os.path.join(results_dir, 'decision_tree.png'))
        
        return graph
    except Exception as e:
        print(f"Error creating decision tree diagram: {e}")
        return None

def main():
    # Create results directory
    results_dir = create_results_dir()
    
    # Define paths - using best practices to find data directory
    # First try the original path
    data_dir = 'crowdfunding-analysis/webscraper/scrapers/scraped_data'
    
    # If that doesn't work, try to build an absolute path based on script location
    if not os.path.exists(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        data_dir = os.path.join(parent_dir, "webscraper", "scrapers", "scraped_data")
        
        # One more alternative if that didn't work
        if not os.path.exists(data_dir):
            data_dir = os.path.join(os.path.dirname(script_dir), "webscraper", "scrapers", "scraped_data")
    
    # Load data
    print("Loading project data...")
    projects = load_project_data(data_dir)
    
    if not projects:
        print("No projects loaded. Please check the data directory path.")
        return
    
    # Output statistics about the loaded data
    live_count = len(projects)
    successful_count = sum(1 for p in projects if p['success'] == 1)
    failed_count = sum(1 for p in projects if p['success'] == 0)
    print(f"Loaded {len(projects)} projects: {live_count} live")
    print(f"Success breakdown: {successful_count} successful/predicted to succeed, {failed_count} failed/predicted to fail")
    
    # Create dataset
    print("Extracting features...")
    features_list, labels, live_projects = create_dataset(projects, for_training=True)
    
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
    for project in features_list:
        # Add features present in the dataset not already in baseline_features
        for key in project.keys():
            if key not in baseline_features and key != 'all_updates_text' and key not in [
                'percent_funded', 'projected_final_percent', 'backers_count', 'backers_per_day'  # Exclude backers_per_day too
            ]:
                baseline_features.append(key)
    
    print(f"Using {len(baseline_features)} features after careful feature selection")
    print(f"Excluded direct predictors: backers_count, backers_per_day")
    print(f"Included more nuanced indicator: avg_pledge_amount")
    
    # Convert features to DataFrame with only our allowed features
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
    
    # Handle class imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y), y=y
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    print(f"Class weights to address imbalance: {class_weight_dict}")
    
    # Skip model training if not enough data
    if len(set(y)) < 2 or len(y) < 10:
        print("Not enough diverse data for model training. Skipping model training.")
        # Just analyze live projects based on time/funding ratio
        feature_names = baseline_features + [f'tfidf_{i}' for i in range(text_features.shape[1])]
        live_projects_analysis = analyze_live_projects(live_projects, None, feature_names)
        
        save_analysis_summary(results_dir, pd.DataFrame({
            'feature': feature_names,
            'importance': [1] * len(feature_names)
        }), "No model trained", None, live_projects_analysis,
        projects_summary={
            'total': len(projects),
            'live': live_count,
            'completed': 0,
            'successful': successful_count,
            'failed': failed_count
        })
        
        # Create visualization of live projects status
        visualize_live_projects_status(live_projects_analysis, results_dir)
            
        print(f"\nResults have been saved to: {results_dir}")
        return
    
    # For model training, use cross-validation
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
    
    # Split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest with class weights
    print("Training Random Forest model with cross-validation...")
    # Use a fully-capable Random Forest with optimal parameters
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
    
    # Calculate various metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    if len(set(y_test)) > 1:  # Only calculate ROC AUC if there are both classes
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC: {roc_auc:.4f}")
    
    print("\nModel Evaluation:")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")
    
    classification_rep = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(classification_rep)
    
    # Get feature importance
    feature_names = baseline_features + [f'tfidf_{i}' for i in range(text_features.shape[1])]
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance.head(10))
    
    # Create and save feature importance visualization
    visualize_feature_importance(importance, results_dir=results_dir)
    
    # Create and save decision tree visualization
    print("\nCreating decision tree diagram...")
    top_features = importance.nlargest(10, 'importance')['feature'].tolist()
    tree_result = create_decision_tree_diagram(
        X_train, y_train, 
        feature_names=top_features,
        results_dir=results_dir
    )
    
    if tree_result is None:
        print("Note: Decision tree visualization was not created.")
    
    # Analyze live projects
    print("\nAnalyzing live projects...")
    live_projects_analysis = analyze_live_projects(live_projects, rf_model, feature_names)
    
    # Visualize live projects status
    visualize_live_projects_status(live_projects_analysis, results_dir=results_dir)
    
    # Save detailed results
    save_analysis_summary(results_dir, importance, classification_rep, tfidf, live_projects_analysis,
                         cv_scores=cv_scores, test_accuracy=accuracy, f1_score=f1, roc_auc=roc_auc,
                         class_weights=class_weight_dict,
                         projects_summary={
                             'total': len(projects),
                             'live': live_count,
                             'completed': 0,
                             'successful': successful_count,
                             'failed': failed_count
                         })
    
    print(f"\nResults have been saved to: {results_dir}")
    
    # Print key findings
    print("\nKey findings:")
    top_features = importance.head(3)['feature'].tolist()
    
    # Create readable labels for top features
    feature_labels = {
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
        'days_left': 'Days Left in Campaign'
    }
    
    # Create readable labels for top features
    readable_features = []
    for feature in top_features:
        if feature in feature_labels:
            readable_features.append(feature_labels[feature])
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
    print(f"2. Cross-validation accuracy: {cv_scores.mean():.1%} (more reliable than test accuracy)")
    
    # Print live project statistics
    likely_success = sum(1 for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to succeed')
    likely_fail = sum(1 for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to fail')
    print(f"3. Live projects: {likely_success} likely to succeed, {likely_fail} likely to fail")
    
    print("\nCheck the results directory for detailed analysis and visualizations in:")
    print(f"- {results_dir}/analysis_summary.txt (Text report)")
    print(f"- {results_dir}/feature_importance.png (Feature importance visualization)")
    if pydotplus_available:
        print(f"- {results_dir}/decision_tree.png (Decision flow diagram)")
    print(f"- {results_dir}/live_projects_status.png (Projects funding status visualization)")

if __name__ == "__main__":
    main() 