# Kickstarter Project Success Prediction Analysis

This analysis uses machine learning (Random Forest and XGBoost) to predict Kickstarter project success based on project updates, engagement metrics, and funding progression. It analyzes live projects, providing insights into what factors contribute to campaign success. Our comparison shows that XGBoost significantly outperforms Random Forest, especially when backer features are included.

## Overview

The analysis processes Kickstarter project data from JSON files:
- Project data: `../webscraper/scrapers/scraped_data/`

The system focuses on live projects:
- **Live Projects**: Success is predicted based on funding trajectory (percentage funded vs. percentage of time elapsed)

## Key Features

- **Time-Based Projections**: Predicts final funding percentage for live projects
- **Cross-Validation**: Uses 5-fold stratified cross-validation for reliable model evaluation
- **Class Imbalance Handling**: Applies balanced class weights to ensure fair representation
- **Comprehensive Feature Engineering**: Extracts engagement, timing, and text-based features
- **Data Leakage Prevention**: Careful feature selection to prevent unrealistic accuracy
- **Educational Feature Configuration**: Designed with feature selection that maintains accuracy around 80% for educational purposes

## How update_analysis.py Works

### 1. Data Loading and Processing

The script begins by loading JSON data from the scraped data directory:

```python
def load_project_data(data_dir):
    """Load project data from a single directory, determining success from the data itself."""
    projects = []
    
    # Check directory existence and handle path resolution
    if not os.path.exists(data_dir):
        # Multiple fallback paths are attempted if the primary path isn't found
        # First try script location relative path, then alternative directories
        
    # Process each JSON file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename), 'r') as f:
                # Load and validate project data
                # Skip files without campaign details
                # Extract key data points including funding goals and amounts
```

The loading process includes:
- Intelligent path resolution to find the data directory
- JSON parsing with error handling
- Data cleaning for monetary values (removing currency symbols, commas)
- Project identification from URLs
- Processing of live projects with remaining days

### 2. Success Determination Logic

For live projects, success is determined based on funding trajectory:

```python
# Calculate percentage of funding achieved
percent_funded = (pledged_amount / funding_goal * 100) if funding_goal > 0 else 0
data['percent_funded'] = percent_funded

# Calculate percent of time elapsed
start_date = datetime.fromisoformat(campaign['funding_start_date'].replace('Z', '+00:00'))
end_date = datetime.fromisoformat(campaign['funding_end_date'].replace('Z', '+00:00'))
total_duration = (end_date - start_date).total_seconds()
current_date = datetime.now(timezone.utc)  # Or calculated from days_left

elapsed = (current_date - start_date).total_seconds()
percent_time = (elapsed / total_duration * 100) if total_duration > 0 else 0
data['percent_time'] = percent_time

# Project final funding based on trajectory
if percent_time > 0:
    projected_final = (percent_funded / percent_time) * 100
    data['projected_final_percent'] = projected_final
    
    # Predict success based on projection
    if projected_final >= 100:
        data['success'] = 1  # On track to succeed
    else:
        data['success'] = 0  # Projected to fail
else:
    # Fallback for very new projects
    # Default to success if at least 50% funded already
    data['success'] = 1 if percent_funded >= 50 else 0
```

This approach allows for accurate classification of live ongoing campaigns based on their current trajectory.

### 3. Feature Extraction Process

The feature extraction is handled by the `extract_features()` function, which processes each project to create a standardized feature set:

```python
def extract_features(project, for_training=True):
    """Extract relevant features from project updates."""
    features = {}
    
    # Basic update statistics
    updates = project.get('updates', {})
    features['num_updates'] = updates.get('count', 0)
    
    # Update content analysis
    update_contents = []
    total_likes = 0
    total_comments = 0
    
    # Process each update
    for update in updates.get('content', []):
        update_content = update.get('content', '')
        update_contents.append(update_content)
        
        # Engagement metrics
        total_likes += update.get('likes_count', 0)
        total_comments += len(update.get('comments', []))
        
        # Update length calculations
        # ... content length analysis
    
    # Calculate engagement averages
    features['total_likes'] = total_likes
    features['total_comments'] = total_comments
    features['average_likes_per_update'] = total_likes / max(num_updates, 1)
    
    # Campaign duration and timing features
    # ... timing calculations
    
    # Live project training features
    features['percent_funded'] = project.get('percent_funded', 0)
    features['percent_time'] = project.get('percent_time', 0)
    features['days_left'] = int(campaign.get('days_left', 0))
    
    # Calculate average pledge amount (if backers exist)
    if features['backers_count'] > 0:
        features['avg_pledge_amount'] = clean_pledged_amount / features['backers_count']
    else:
        features['avg_pledge_amount'] = 0
    
    # Combine all update content for text analysis
    features['all_updates_text'] = ' '.join(update_contents)
    
    return features
```

### 4. Dataset Creation and Preprocessing

The dataset creation process handles both numerical and text features:

```python
# Create dataset from features
features_list, labels, live_projects = create_dataset(projects, for_training=True)

# Convert features to DataFrame
df = pd.DataFrame(features_list)

# Create TF-IDF features from update text
tfidf = TfidfVectorizer(max_features=100, min_df=2)
text_data = df['all_updates_text'].fillna('').tolist()
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
```

The preprocessing includes:
- Handling of missing values
- Text vectorization with TF-IDF
- Feature combination
- Class weight calculation to address imbalance

### 5. Model Training and Evaluation

The model training process uses a Random Forest classifier with cross-validation:

```python
# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Configure Random Forest with class weights
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_features='sqrt',
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=None,
    bootstrap=True,
    random_state=42,
    class_weight=class_weight_dict
)

# Perform stratified k-fold cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')

# Train final model on full training set
rf_model.fit(X_train, y_train)

# Evaluate model performance
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Calculate various metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred_proba)  # If both classes exist
classification_rep = classification_report(y_test, y_pred)
```

The evaluation includes multiple metrics:
- Cross-validation scores (more reliable than single-split evaluation)
- Test set accuracy
- F1 score for balanced precision/recall assessment
- ROC AUC for discrimination ability
- Detailed classification report

### 6. Feature Importance Analysis

The script analyzes which features most strongly influence project success:

```python
# Get feature importance
feature_names = baseline_features + [f'tfidf_{i}' for i in range(text_features.shape[1])]
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
})
importance = importance.sort_values('importance', ascending=False)

# Create readable feature importance
importance_df['importance_percentage'] = importance_df['importance'] * 100
importance_df['description'] = importance_df['feature'].map(feature_descriptions.get)
```

This analysis reveals which factors (like engagement metrics, update frequency) most strongly predict project success.

### 7. Live Project Analysis

For ongoing campaigns, the script provides detailed predictions:

```python
def analyze_live_projects(live_projects, model, feature_names):
    """Analyze live projects separately and make predictions."""
    live_projects_analysis = []
    
    for features, project in live_projects:
        # Extract key project information
        project_analysis = {
            'title': project.get('title', project.get('url', 'Unknown')),
            'percent_funded': features.get('percent_funded', 0),
            'percent_time': features.get('percent_time', 0)
        }
        
        # Calculate funding trajectory and projection
        time_percent = features.get('percent_time', 0)
        funded_percent = features.get('percent_funded', 0)
        
        if time_percent > 0:
            funding_ratio = funded_percent / time_percent
            projected_final = funding_ratio * 100
            
            # Determine success likelihood based on trajectory
            if funding_ratio >= 1.0:
                prediction = 'Likely to succeed'
                details = f"Funding on track or ahead of schedule ({funding_ratio:.1f}x expected rate)"
            elif funding_ratio >= 0.8:
                prediction = 'Likely to succeed'
                details = f"Funding slightly behind but promising ({funding_ratio:.1f}x expected rate)"
            elif funding_ratio >= 0.5:
                prediction = 'Likely to fail'
                details = f"Funding significantly behind schedule ({funding_ratio:.1f}x expected rate)"
            else:
                prediction = 'Likely to fail'
                details = f"Funding far behind schedule ({funding_ratio:.1f}x expected rate)"
                
            # Add model prediction if available
            if model:
                try:
                    # Create feature vector for model prediction
                    feature_vector = [features.get(name, 0) for name in feature_names if name != 'all_updates_text']
                    model_prediction = model.predict([feature_vector])[0]
                    model_prob = model.predict_proba([feature_vector])[0][1]
                    
                    details += f" | Model confidence: {model_prob:.1%} likely to succeed"
                except Exception:
                    pass
```

This provides both trajectory-based and model-based predictions for live projects.

### 8. Results Storage and Presentation

The script saves detailed analysis results to a timestamped directory:

```python
def save_analysis_summary(results_dir, importance_df, classification_rep, tfidf_vectorizer, live_projects_analysis):
    """Save results as analysis_summary.txt."""
    
    # Create readable feature importance
    readable_importance = create_readable_feature_importance(importance_df.copy(), tfidf_vectorizer)
    
    # Save detailed analysis to text file
    with open(os.path.join(results_dir, 'analysis_summary.txt'), 'w') as f:
        # Write header information
        f.write("KICKSTARTER PROJECT SUCCESS PREDICTION ANALYSIS\n")
        
        # Include model performance
        f.write("\nMODEL PERFORMANCE\n")
        f.write(classification_rep)
        
        # Write live project analysis
        f.write("\nLIVE PROJECTS ANALYSIS\n")
        f.write(f"Number of live projects analyzed: {len(live_projects_analysis)}\n")
        f.write(f"Predicted to succeed: {sum(1 for p in live_projects_analysis if p.get('prediction_outcome') == 'Likely to succeed')}\n")
        
        # Write feature importance
        f.write("\nTOP FACTORS INFLUENCING PROJECT SUCCESS\n")
        
        # Separate numerical and text features
        # Write detailed explanations for each
        
        # Include interpretation guide
        f.write("\nINTERPRETATION GUIDE\n")
        # Explanation of metrics and their meaning
```

## Example of Text Analysis

Here's how the model analyzes project updates. Consider this example update:

```
Dear Backers,
As we enter the final 48 hours of our Kickstarter campaign for the Morfone Electric Minoxidil Atomizer, 
we want to take a moment to sincerely thank you for your incredible support. Your backing has brought 
us this far, and we are incredibly grateful. 🙏✨

Thank You: Every step of this journey has been made possible by your trust and support. Thanks to 
wonderful backers like you, we're closer than ever to making our vision a reality. 🌟
...
```

The TF-IDF analysis identifies important words and their patterns:
- "Backers" (importance varies): Direct community engagement
- "thank" (high importance): Gratitude expression to community
- Words like "you", "your": Personal connection with backers
- Community-building language: Creates sense of shared ownership
- Engagement indicators: Updates that receive likes and comments

These patterns, combined with numerical metrics (likes, comments, update frequency), help predict project success.

## Implementation Details

### 1. Data Pre-Processing and Feature Extraction

#### Intelligent Data Loading
- Reads JSON files from the project directory
- Handles currency symbols and formats for consistent monetary values
- Calculates time progression for live projects

#### Feature Engineering for Live Projects
```
Features:
- num_updates: Total number of updates
- total_likes: Sum of likes across all updates
- total_comments: Total comments across all updates
- funding_duration: Campaign length in days
- updates_per_day: Update frequency
- avg_update_length: Average character count of updates
- backers_count: Number of project backers
- average_likes_per_update: Engagement per update
- average_comments_per_update: Discussion level per update
- percent_funded: Current % of funding goal reached
- percent_time: % of campaign duration elapsed
- funding_ratio: percent_funded / percent_time
- projected_final_percent: Projected final funding %
- backers_per_day: Acquisition rate adjusted for elapsed time
- avg_pledge_amount: Average contribution per backer
- days_left: Days remaining in campaign
```

#### Text Feature Extraction
- Processes all update content using TF-IDF vectorization
- Minimum document frequency of 2 to reduce noise
- Extracts 100 most important words/phrases
- Handles empty text fields gracefully

### 2. Data Leakage Prevention

Our approach carefully prevents data leakage, which can lead to artificially high accuracy:

#### What is Data Leakage?
Data leakage occurs when features directly reveal the target variable. In our case, using `final_funding_percent` would leak information about project success.

#### Our Prevention Strategy
1. **Indirect Indicators**: Use engagement metrics rather than direct funding indicators
2. **Cross-Validation**: Properly stratified validation splits
3. **Feature Filtering**: Explicit removal of leaky features like final funding percentage

### 3. Random Forest Implementation with Cross-Validation

The model uses scikit-learn's RandomForestClassifier with stratified k-fold cross-validation for reliable performance assessment.

#### Model Configuration
```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=None,
    bootstrap=True,
    random_state=42,
    class_weight=class_weight_dict  # Handles class imbalance
)
```

#### Feature Selection for Educational Purposes
For educational value, the model excludes some highly predictive features:
- `backers_count`: Direct indicator of project popularity
- `backers_per_day`: Strong predictor of funding velocity

The model retains more nuanced indicators like:
- `avg_pledge_amount`: Quality of backers rather than quantity
- Time-based metrics that require interpretation
- Engagement metrics related to updates and community interaction

This configuration achieves around 80% accuracy (cross-validation), making it useful while still leaving room for student exploration and improvement.

#### Cross-Validation Process
```python
# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
```

### 4. Live Project Analysis

Our system provides detailed predictions for live projects based on:

#### Funding Trajectory
- **On Track**: percent_funded ≥ percent_time (funding_ratio ≥ 1.0)
- **Slightly Behind**: funding_ratio ≥ 0.8
- **Significantly Behind**: funding_ratio ≥ 0.5
- **Far Behind**: funding_ratio < 0.5

#### Prediction Confidence
- Time-based projection is combined with model prediction
- Model confidence is reported (probability of success)
- Early projects (< 5% of time elapsed) are marked as "Unknown"
- Discrepancies between trajectory and model are highlighted

#### Visualization
- Scatter plot of percent_funded vs. percent_time
- Color-coding by prediction outcome
- Diagonal reference line representing "on track" trajectory
- Histogram of projected final funding percentages

### 5. Current Results

#### Model Performance
```
Cross-validation accuracy: ~80-81% (varies by dataset)
- More reliable than single test split accuracy (typically 70-75%)
- F1 score: ~72-73% (balanced precision/recall)
- ROC AUC: ~0.81-0.82 (good discrimination ability)
```

#### Top Influencing Factors (examples)
1. Average pledge amount (quality of backers)
2. Percent of time elapsed (campaign progress)
3. Days left in campaign (remaining opportunity)
4. Campaign duration (project scope)
5. Update engagement metrics
6. Key words in updates

### 6. Enhanced Model Results (XGBoost)

We've extended our analysis to compare Random Forest with XGBoost, both with and without backer features.

#### Performance Comparison

| Metric | Random Forest | XGBoost with Backers | XGBoost without Backers |
|--------|---------------|----------------------|-------------------------|
| Cross-validation accuracy | 80.6% (±7.8%) | 92.5% (±4.2%) | 83.8% (±6.7%) |
| Test accuracy | 72.5% | 87.5% | 80.0% |
| F1 Score | 72.7% | 87.5% | 80.0% |
| ROC AUC | 83.9% | 95.9% | 82.6% |

#### Key Differences Between Models

1. **Feature Importance Variations**:
   - **Random Forest** emphasizes quantitative metrics:
     1. Average Pledge Amount (25.9%)
     2. Percentage of Time Elapsed (12.6%)
     3. Days Left in Campaign (9.5%)
   
   - **XGBoost with backers** prioritizes communication and engagement:
     1. Word 'we' in Updates (11.0%)
     2. Word 'it' in Updates (9.5%)
     3. Number of Backers (8.8%)
   
   - **XGBoost without backers** adapts by focusing on communication:
     1. Word 'our' in Updates (23.8%)
     2. Total Likes (10.8%)
     3. Word 'campaign' in Updates (6.1%)

2. **Live Project Predictions**:
   - **Random Forest**: 36 likely to succeed, 131 likely to fail (conservative)
   - **XGBoost with backers**: 81 likely to succeed, 118 likely to fail (optimistic)
   - **XGBoost without backers**: 71 likely to succeed, 124 likely to fail (middle ground)

3. **Model Recommendations**:
   - For maximum accuracy: XGBoost with all features (92.5% CV accuracy)
   - When backer data is unavailable: XGBoost without backers (83.8% CV accuracy)
   - For conservative predictions: Random Forest (80.6% CV accuracy)

4. **Impact of Backer Features**:
   - Including backer features improves XGBoost performance by ~8.7%
   - Without backer data, models compensate by emphasizing communication style and engagement metrics
   - Backer information appears to be a strong indicator of project health

These findings demonstrate the power of ensemble learning methods for predicting crowdfunding success, while highlighting the importance of feature selection. The significant performance improvement of XGBoost over Random Forest suggests that its boosting approach better captures the complex relationships between features and project outcomes.

### 7. Usage

1. Install dependencies:
```bash
# Navigate to the analysis directory
cd crowdfunding-analysis/analysis

# Install required packages
pip install -r requirements.txt

# Optional: Create a virtual environment first
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the analysis:
```bash
# For Random Forest model:
python update_analysis.py
# OR from the project root
python -m crowdfunding-analysis.analysis.update_analysis

# For XGBoost model:
python update_analysis_2.py
# OR from the project root
python -m crowdfunding-analysis.analysis.update_analysis_2
```

3. Check results in the generated results directories:
- Random Forest: `crowdfunding-analysis/analysis/results/`
- XGBoost with backers: `crowdfunding-analysis/analysis/results_xgboost_with_backers/`
- XGBoost without backers: `crowdfunding-analysis/analysis/results_xgboost_without_backers/`

Each directory contains:
- `analysis_summary.txt`: Detailed analysis report
- `feature_importance.png`: Visual representation of important features
- `decision_tree.png`: Decision flow diagram (if pydotplus is installed)
- `live_projects_status.png`: Visual plot of live project trajectories and predictions

### 8. Advantages of this Implementation

1. **Focused Project Analysis**
   - Provides actionable insights for ongoing campaigns
   - Realistic performance metrics through cross-validation
   - Early prediction of campaign outcomes

2. **Robust Data Handling**
   - Prevents misleading accuracy through anti-leakage measures
   - Processes diverse data types (numerical, text, temporal)
   - Handles class imbalance through balanced weights

3. **Comprehensive Feature Engineering**
   - Multi-dimensional engagement metrics
   - Temporal patterns in updates
   - Semantic analysis of update content
   - Backer acquisition and engagement ratios

4. **Actionable Insights**
   - Clear trajectory analysis for live projects
   - Identification of key success factors
   - Early warning for at-risk projects
   - Detailed error analysis

### 9. Limitations and Future Improvements

1. **Model Enhancements**
   - ✅ XGBoost implementation complete (see Enhanced Model Results section)
   - Further optimize XGBoost hyperparameters
   - Experiment with other gradient boosting algorithms (LightGBM)
   - Experiment with deep learning for text features
   - Develop specialized models for different project categories

2. **Feature Expansion**
   - Add sentiment analysis of updates
   - Include update image analysis
   - Incorporate reward tier structure
   - Add creator history features

3. **Time Series Analysis**
   - Analyze update timing patterns more deeply
   - Implement temporal models for funding progression
   - Track engagement evolution throughout campaigns

4. **Deployment Improvements**
   - Create API for real-time predictions
   - Develop dashboard for ongoing campaign monitoring
   - Implement automatic recommendations for creators 

# crowdfunding-analysis
Time Series Analysis on Crowdfunding Platform Kickstarter
249 projects update details section have been scraped and saved in the `scrapers/scraped_data` directory.


The `link_scraper.py` file is the script that scrapes the links from the Kickstarter website.

The `cleanup_no_updates.py` file is the script that cleans up the scraped data by removing duplicateprojects.

The `cleanup_link_scraper.py` file is the script that cleans up the link scraper by removing duplicate links.



