import json
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import re
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import class_weight

def load_project_data(data_dir):
    """Load project data from a single directory, determining success from the data itself."""
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
    # Load all projects from the single directory
    file_count = 0
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            file_count += 1
            with open(os.path.join(data_dir, filename), 'r') as f:
                try:
                    data = json.load(f)
                    
                    # Skip if no campaign details
                    if 'campaign_details' not in data or not data['campaign_details']:
                        print(f"Skipping {filename} - no campaign details")
                        continue
                    
                    # Add title for better project identification in reports
                    if 'url' in data:
                        # Extract project name from URL
                        project_name = data['url'].split('/')[-1]
                        data['title'] = project_name
                    
                    campaign = data['campaign_details']
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
                    
                    # Determine if project is live or completed
                    is_live = False
                    if 'days_left' in campaign:
                        is_live = True
                        data['is_live'] = True
                    else:
                        data['is_live'] = False

                    # Calculate success differently for live vs completed projects
                    if is_live:
                        # For live projects, calculate percentage of funding and time
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
                    else:
                        # For completed projects - simple success metric
                        is_successful = funding_goal > 0 and pledged_amount >= funding_goal
                        data['success'] = 1 if is_successful else 0
                        data['success_prediction'] = f"Completed: {(pledged_amount / funding_goal * 100):.1f}% funded"
                    
                    projects.append(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {filename}, skipping file")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}, skipping file")
    
    print(f"Found {file_count} JSON files, loaded {len(projects)} projects successfully")
    return projects

def extract_features(project, for_training=True):
    """Extract relevant features from project updates.
    
    Args:
        project: The project data
        for_training: If True, prepare features for model training. If False, prepare for prediction.
    """
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
    
    # Create features that won't create data leakage
    is_live = project.get('is_live', False)
    features['is_live'] = 1 if is_live else 0
    
    if for_training:
        # For training, we need to handle live and completed projects differently
        if is_live:
            # For live projects, include projection metrics
            features['percent_funded'] = project.get('percent_funded', 0)
            features['percent_time'] = project.get('percent_time', 0)
            if 'projected_final_percent' in project:
                features['projected_final_percent'] = project.get('projected_final_percent', 0)
        else:
            # For completed projects in training, we still want to include success-related 
            # metrics, but more carefully to avoid direct data leakage
            
            # Instead of direct funding percentage (which is too close to the target),
            # we'll use backer engagement metrics and other indirect indicators
            clean_funding_goal = project.get('clean_funding_goal', 0)
            clean_pledged_amount = project.get('clean_pledged_amount', 0)
            
            # Create ratio features that aren't direct indicators of success
            features['backers_per_day'] = features['backers_count'] / max(funding_duration, 1)
            features['avg_pledge_amount'] = clean_pledged_amount / max(features['backers_count'], 1)
            features['has_updates'] = 1 if features['num_updates'] > 0 else 0
            
            # Frequency of updates relative to duration
            if funding_duration > 0:
                features['update_frequency'] = features['num_updates'] / funding_duration
            else:
                features['update_frequency'] = 0
            
            # Calculate ratio of likes to backers (engagement metric)
            if features['backers_count'] > 0:
                features['likes_per_backer'] = total_likes / features['backers_count']
            else:
                features['likes_per_backer'] = 0
    else:
        # For prediction on new projects, use all available metrics
        if is_live:
            # For live projects
            features['percent_funded'] = project.get('percent_funded', 0)
            features['percent_time'] = project.get('percent_time', 0)
            features['funding_ratio'] = features['percent_funded'] / max(features['percent_time'], 1)
            
            # Calculate projected final
            if 'percent_time' in project and project['percent_time'] > 0:
                projected_final = project['percent_funded'] / project['percent_time'] * 100
                features['projected_final_percent'] = projected_final
            else:
                features['projected_final_percent'] = 0
                
            # Calculate backers metrics
            features['backers_per_day'] = features['backers_count'] / max(funding_duration * (features['percent_time']/100), 1)
        else:
            # For completed projects
            features['backers_per_day'] = features['backers_count'] / max(funding_duration, 1)
            features['avg_pledge_amount'] = project.get('clean_pledged_amount', 0) / max(features['backers_count'], 1)
            
            if features['backers_count'] > 0:
                features['likes_per_backer'] = total_likes / features['backers_count']
            else:
                features['likes_per_backer'] = 0
    
    # Combine all update content for text analysis
    features['all_updates_text'] = ' '.join(update_contents)
    
    return features

def create_dataset(projects, for_training=True):
    """Create a dataset from the project features.
    
    Args:
        projects: List of project data
        for_training: If True, prepare for model training; if False, for prediction
    """
    features_list = []
    labels = []
    live_projects = []
    completed_projects = []
    
    for project in projects:
        features = extract_features(project, for_training=for_training)
        
        # Separate live and completed projects
        if project.get('is_live', False):
            live_projects.append((features, project))
        else:
            completed_projects.append((features, project))
        
        features_list.append(features)
        labels.append(project['success'])
    
    # Print dataset statistics
    print(f"Dataset created with {len(features_list)} projects: {len(live_projects)} live, {len(completed_projects)} completed")
    print(f"Class distribution: {sum(labels)} successful, {len(labels) - sum(labels)} unsuccessful")
    
    return features_list, labels, live_projects, completed_projects

def create_results_dir():
    """Create a results directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f'crowdfunding-analysis/analysis/results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def create_readable_feature_importance(importance_df, tfidf_vectorizer=None):
    """Create a more readable feature importance analysis."""
    # Convert importance to percentage
    importance_df['importance_percentage'] = importance_df['importance'] * 100
    
    # Create readable descriptions for features
    feature_descriptions = {
        'funding_duration': 'Campaign Duration (days)',
        'total_likes': 'Total Likes on Updates',
        'total_comments': 'Total Comments on Updates',
        'total_comment_count': 'Total Comment Count',
        'num_updates': 'Number of Project Updates',
        'updates_per_day': 'Updates Posted per Day',
        'avg_update_length': 'Average Length of Updates',
        'is_live': 'Project is Currently Live',
        'percent_funded': 'Percentage of Funding Goal Reached (Live)',
        'percent_time': 'Percentage of Campaign Duration Elapsed (Live)',
        'projected_final_percent': 'Projected Final Funding Percentage (Live)',
        'final_funding_percent': 'Final Funding Percentage (Completed)'
    }
    
    # Add descriptions for TF-IDF features if vectorizer is provided
    if tfidf_vectorizer:
        feature_terms = {f"tfidf_{i}": term for term, i in tfidf_vectorizer.vocabulary_.items()}
        feature_descriptions.update(feature_terms)
    
    # Add readable descriptions
    importance_df['description'] = importance_df['feature'].map(lambda x: feature_descriptions.get(x, x))
    
    # Format the importance percentage
    importance_df['importance_formatted'] = importance_df['importance_percentage'].map('{:.2f}%'.format)
    
    return importance_df[['feature', 'description', 'importance_formatted', 'importance_percentage']]

def save_readable_results(results_dir, importance_df, classification_rep, tfidf_vectorizer, live_projects_analysis):
    """Save results in a more readable format."""
    # Create readable feature importance
    readable_importance = create_readable_feature_importance(importance_df.copy(), tfidf_vectorizer)
    
    # Save detailed analysis
    with open(os.path.join(results_dir, 'analysis_summary.txt'), 'w') as f:
        f.write("KICKSTARTER PROJECT SUCCESS PREDICTION ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 20 + "\n")
        f.write(classification_rep)
        f.write("\n\n")
        
        f.write("LIVE PROJECTS ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Number of live projects analyzed: {len(live_projects_analysis)}\n")
        f.write(f"Predicted to succeed: {sum(1 for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to succeed')}\n")
        f.write(f"Predicted to fail: {sum(1 for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to fail')}\n\n")
        
        # Add example predictions for some live projects
        f.write("EXAMPLE LIVE PROJECT PREDICTIONS:\n")
        for i, project in enumerate(live_projects_analysis[:5]):
            f.write(f"\n{i+1}. {project.get('title', 'Unnamed Project')}\n")
            f.write(f"Current Status: {project.get('funding_status', '')}\n")
            f.write(f"Prediction: {project.get('prediction_outcome', '')}\n")
            f.write(f"Details: {project.get('prediction_details', '')}\n")
        
        f.write("\n\nTOP FACTORS INFLUENCING PROJECT SUCCESS\n")
        f.write("-" * 35 + "\n")
        top_features = readable_importance.head(15)
        
        # Separate numerical and text features
        numerical_features = top_features[top_features['feature'].isin([
            'funding_duration', 'total_likes', 'total_comments', 
            'total_comment_count', 'num_updates', 'updates_per_day',
            'avg_update_length', 'is_live', 'percent_funded', 'percent_time',
            'projected_final_percent', 'final_funding_percent'
        ])]
        text_features = top_features[~top_features['feature'].isin([
            'funding_duration', 'total_likes', 'total_comments', 
            'total_comment_count', 'num_updates', 'updates_per_day',
            'avg_update_length', 'is_live', 'percent_funded', 'percent_time',
            'projected_final_percent', 'final_funding_percent'
        ])]
        
        f.write("\nNUMERICAL FACTORS:\n")
        for _, row in numerical_features.iterrows():
            f.write(f"\n{row['description']}\n")
            f.write(f"Impact on Success: {row['importance_formatted']}\n")
        
        f.write("\nCOMMUNICATION PATTERNS IN UPDATES:\n")
        f.write("The following words/phrases in project updates are associated with higher success rates:\n")
        for _, row in text_features.iterrows():
            word = row['description'].replace('tfidf_', '').strip()
            f.write(f"\n{word}\n")
            f.write(f"Impact on Success: {row['importance_formatted']}\n")
        
        f.write("\n\nINTERPRETATION GUIDE\n")
        f.write("-" * 20 + "\n")
        f.write("1. The percentages show how much each factor contributes to the model's decision-making.\n")
        f.write("2. Higher percentages indicate stronger influence on project success.\n")
        f.write("3. For numerical factors (like duration, likes), higher values generally indicate better chances of success.\n")
        f.write("4. For live projects, the prediction is based on current funding progress vs. time elapsed.\n")
        f.write("5. For communication patterns, the presence of these words often indicates:\n")
        f.write("   - Direct engagement with backers\n")
        f.write("   - Community-building language\n")
        f.write("   - Regular project updates\n")
        f.write("   - Personal and inclusive communication style\n")
    
    # Save detailed feature importance
    readable_importance.to_csv(os.path.join(results_dir, 'feature_importance_readable.csv'), index=False)
    
    # Save live projects analysis as CSV
    live_df = pd.DataFrame(live_projects_analysis)
    if not live_df.empty:
        live_df.to_csv(os.path.join(results_dir, 'live_projects_predictions.csv'), index=False)

def plot_feature_importance(importance_df, results_dir):
    """Plot and save feature importance visualization."""
    # Create more readable labels
    plot_data = importance_df.head(10).copy()
    feature_labels = {
        'funding_duration': 'Campaign Duration',
        'total_likes': 'Total Likes',
        'total_comments': 'Total Comments',
        'total_comment_count': 'Comment Count',
        'num_updates': 'Number of Updates',
        'updates_per_day': 'Updates per Day',
        'avg_update_length': 'Avg Update Length',
        'is_live': 'Project is Live',
        'percent_funded': '% Funding Reached',
        'percent_time': '% Time Elapsed',
        'projected_final_percent': 'Projected Final %',
        'final_funding_percent': 'Final Funding %'
    }
    
    plot_data['feature'] = plot_data['feature'].map(lambda x: feature_labels.get(x, x))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=plot_data, x='importance', y='feature')
    plt.title('Top 10 Factors Influencing Project Success', pad=20)
    plt.xlabel('Importance Score (higher = more influential)')
    plt.ylabel('Factor')
    
    # Add percentage labels
    for i, v in enumerate(plot_data['importance']):
        plt.text(v, i, f' {v:.1%}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_test, y_pred, results_dir):
    """Plot and save confusion matrix visualization."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

def plot_live_projects_status(live_projects_analysis, results_dir):
    """Plot and save live projects status visualization."""
    if not live_projects_analysis:
        return
    
    # Create DataFrame
    df = pd.DataFrame(live_projects_analysis)
    
    if 'percent_funded' in df.columns and 'percent_time' in df.columns:
        plt.figure(figsize=(10, 8))
        
        # Scatter plot with color coding by prediction
        colors = df['prediction_outcome'].map({
            'Likely to succeed': 'green',
            'Likely to fail': 'red',
            'Unknown': 'gray'
        })
        
        plt.scatter(df['percent_time'], df['percent_funded'], c=colors, alpha=0.6)
        
        # Add diagonal line representing "on track"
        max_val = max(df['percent_time'].max(), df['percent_funded'].max(), 100) * 1.1
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
        
        plt.xlabel('Percentage of Campaign Time Elapsed')
        plt.ylabel('Percentage of Funding Goal Reached')
        plt.title('Live Projects: Funding Progress vs Time Elapsed')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Likely to succeed'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Likely to fail'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Unknown'),
            Line2D([0], [0], linestyle='--', color='k', label='On Track Line')
        ]
        plt.legend(handles=legend_elements)
        
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(results_dir, 'live_projects_status.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a histogram of projected final percentages
        if 'projected_final_percent' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['projected_final_percent'].clip(0, 200), bins=20)
            plt.axvline(x=100, color='r', linestyle='--')
            plt.xlabel('Projected Final Funding Percentage')
            plt.ylabel('Number of Projects')
            plt.title('Distribution of Projected Final Funding Percentages for Live Projects')
            plt.savefig(os.path.join(results_dir, 'projected_funding_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()

def analyze_live_projects(live_projects, model, feature_names):
    """Analyze live projects separately and make predictions."""
    live_projects_analysis = []
    
    for features, project in live_projects:
        project_analysis = {
            'title': project.get('title', project.get('url', 'Unknown')),
            'url': project.get('url', 'Unknown'),
            'percent_funded': features.get('percent_funded', 0),
            'percent_time': features.get('percent_time', 0)
        }
        
        # Funding status description
        project_analysis['funding_status'] = f"{features.get('percent_funded', 0):.1f}% funded at {features.get('percent_time', 0):.1f}% of time"
        
        # Calculate expected vs actual funding for its stage
        time_percent = features.get('percent_time', 0)
        funded_percent = features.get('percent_funded', 0)
        
        # Calculate ratio of funding to time
        if time_percent > 0:
            funding_ratio = funded_percent / time_percent
            projected_final = funding_ratio * 100
            project_analysis['projected_final_percent'] = projected_final
            
            # Set thresholds for prediction
            if funding_ratio >= 1.0:
                prediction = 'Likely to succeed'
                if funding_ratio >= 2.0:
                    details = f"Funding far ahead of schedule ({funding_ratio:.1f}x expected rate)"
                else:
                    details = f"Funding on track or ahead of schedule ({funding_ratio:.1f}x expected rate)"
            elif funding_ratio >= 0.8:
                prediction = 'Likely to succeed'
                details = f"Funding slightly behind schedule but still promising ({funding_ratio:.1f}x expected rate)"
            elif funding_ratio >= 0.5:
                prediction = 'Likely to fail'
                details = f"Funding significantly behind schedule ({funding_ratio:.1f}x expected rate)"
            else:
                prediction = 'Likely to fail'
                details = f"Funding far behind schedule ({funding_ratio:.1f}x expected rate)"
                
            # Override for very early projects
            if time_percent < 5:
                prediction = 'Unknown'
                details = "Too early to predict accurately"
                
            # Add model prediction if possible
            try:
                feature_vector = [features.get(name, 0) for name in feature_names if name != 'all_updates_text']
                # Add TF-IDF features with zeros if needed
                while len(feature_vector) < len(feature_names) - 1:  # -1 for 'all_updates_text'
                    feature_vector.append(0)
                    
                model_prediction = model.predict([feature_vector])[0]
                model_prob = model.predict_proba([feature_vector])[0][1]
                
                details += f" | Model confidence: {model_prob:.1%} likely to succeed"
                
                # Update prediction if model strongly disagrees
                if model_prediction != (1 if prediction == 'Likely to succeed' else 0) and abs(model_prob - 0.5) > 0.25:
                    details += " (model prediction differs from time-based projection)"
            except Exception as e:
                print(f"Error making model prediction: {str(e)}")
                
            project_analysis['prediction_outcome'] = prediction
            project_analysis['prediction_details'] = details
        else:
            # Can't make time-based prediction
            project_analysis['prediction_outcome'] = 'Unknown'
            project_analysis['prediction_details'] = "Unable to determine campaign progress"
        
        # Add campaign_details for reference
        if 'campaign_details' in project:
            for k, v in project.get('campaign_details', {}).items():
                project_analysis[f'campaign_{k}'] = v
        
        live_projects_analysis.append(project_analysis)
    
    return live_projects_analysis

def main():
    # Create results directory
    results_dir = create_results_dir()
    
    # Define paths - using best practices to find data directory
    # First try the original path
    data_dir = 'crowdfunding-analysis/webscraper/scrapers/scraped_data'
    
    # If that doesn't work, try to build an absolute path based on script location
    if not os.path.exists(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels
        data_dir = os.path.join(project_root, "webscraper", "scrapers", "scraped_data")
        
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
    live_count = sum(1 for p in projects if p.get('is_live', False))
    completed_count = len(projects) - live_count
    successful_count = sum(1 for p in projects if p['success'] == 1)
    failed_count = sum(1 for p in projects if p['success'] == 0)
    print(f"Loaded {len(projects)} projects: {live_count} live, {completed_count} completed")
    print(f"Success breakdown: {successful_count} successful/predicted to succeed, {failed_count} failed/predicted to fail")
    
    # Create dataset
    print("Extracting features...")
    features_list, labels, live_projects, completed_projects = create_dataset(projects, for_training=True)
    
    # Define baseline features that all projects have
    baseline_features = [
        'num_updates', 'total_likes', 'total_comments', 
        'total_comment_count', 'funding_duration', 'updates_per_day',
        'avg_update_length', 'is_live', 'backers_count',
        'has_first_day_update', 'has_first_week_update',
        'average_likes_per_update', 'average_comments_per_update'
    ]
    
    # Add project-specific features
    for project in features_list:
        # Add features present in the dataset not already in baseline_features
        for key in project.keys():
            if key not in baseline_features and key != 'all_updates_text' and key not in [
                'percent_funded', 'percent_time', 'projected_final_percent',
                'final_funding_percent'  # Removed to prevent data leakage
            ]:
                baseline_features.append(key)
    
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
        
        # Create basic visualizations
        plot_live_projects_status(live_projects_analysis, results_dir)
        
        # Save simplified results
        with open(os.path.join(results_dir, 'analysis_summary.txt'), 'w') as f:
            f.write("KICKSTARTER PROJECT SUCCESS PREDICTION ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write("Not enough diverse data for model training.\n\n")
            
            f.write("LIVE PROJECTS ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of live projects analyzed: {len(live_projects_analysis)}\n")
            f.write(f"Predicted to succeed: {sum(1 for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to succeed')}\n")
            f.write(f"Predicted to fail: {sum(1 for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to fail')}\n\n")
            
            # Add example predictions for some live projects
            f.write("EXAMPLE LIVE PROJECT PREDICTIONS:\n")
            for i, project in enumerate(live_projects_analysis[:5]):
                f.write(f"\n{i+1}. {project.get('title', 'Unnamed Project')}\n")
                f.write(f"Current Status: {project.get('funding_status', '')}\n")
                f.write(f"Prediction: {project.get('prediction_outcome', '')}\n")
                f.write(f"Details: {project.get('prediction_details', '')}\n")
        
        # Save live projects analysis as CSV
        live_df = pd.DataFrame(live_projects_analysis)
        if not live_df.empty:
            live_df.to_csv(os.path.join(results_dir, 'live_projects_predictions.csv'), index=False)
            
        print(f"\nResults have been saved to: {results_dir}")
        return
    
    # Use cross-validation for more reliable accuracy assessment
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
    
    # Split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest with class weights
    print("Training Random Forest model with cross-validation...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight=class_weight_dict
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
    
    # Analyze live projects
    print("\nAnalyzing live projects...")
    live_projects_analysis = analyze_live_projects(live_projects, rf_model, feature_names)
    
    # Create visualizations and save results
    print("\nSaving results and creating visualizations...")
    plot_feature_importance(importance, results_dir)
    plot_confusion_matrix(y_test, y_pred, results_dir)
    plot_live_projects_status(live_projects_analysis, results_dir)
    save_readable_results(results_dir, importance, classification_rep, tfidf, live_projects_analysis)
    
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
        'is_live': 'Project is Live',
        'percent_funded': 'Percentage Funded',
        'percent_time': 'Percentage of Time Elapsed',
        'projected_final_percent': 'Projected Final Percentage',
        'backers_count': 'Number of Backers',
        'backers_per_day': 'Backers per Day',
        'average_likes_per_update': 'Average Likes per Update',
        'average_comments_per_update': 'Average Comments per Update'
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
    
    # Add information about completed projects
    completed_success = sum(1 for _, p in completed_projects if p.get('success') == 1)
    completed_fail = sum(1 for _, p in completed_projects if p.get('success') == 0)
    print(f"4. Completed projects: {completed_success} successful, {completed_fail} failed")
    
    print("\nCheck the results directory for detailed analysis and visualizations.")

if __name__ == "__main__":
    main() 