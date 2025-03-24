# Kickstarter Project Success Prediction Analysis

This analysis uses machine learning (Random Forest) to predict Kickstarter project success based on project updates, engagement metrics, and funding progression. It analyzes both completed and live projects, providing insights into what factors contribute to campaign success.

## Overview

The analysis processes Kickstarter project data from JSON files:
- Project data: `../webscraper/scrapers/scraped_data/`

The system intelligently handles two distinct types of projects:
- **Completed Projects**: Success is determined by whether the pledged amount meets or exceeds the funding goal
- **Live Projects**: Success is predicted based on funding trajectory (percentage funded vs. percentage of time elapsed)

## Key Features

- **Dual Analysis System**: Separate analysis paths for live and completed projects
- **Time-Based Projections**: Predicts final funding percentage for live projects
- **Cross-Validation**: Uses 5-fold stratified cross-validation for reliable model evaluation
- **Class Imbalance Handling**: Applies balanced class weights to ensure fair representation
- **Comprehensive Feature Engineering**: Extracts engagement, timing, and text-based features
- **Data Leakage Prevention**: Careful feature selection to prevent unrealistic accuracy

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
- Automatically detects and classifies live vs. completed projects
- Handles currency symbols and formats for consistent monetary values
- Calculates time progression for live projects

#### Feature Engineering for All Projects
```
Common Base Features:
- num_updates: Total number of updates
- total_likes: Sum of likes across all updates
- total_comments: Total comments across all updates
- funding_duration: Campaign length in days
- updates_per_day: Update frequency
- avg_update_length: Average character count of updates
- backers_count: Number of project backers
- average_likes_per_update: Engagement per update
- average_comments_per_update: Discussion level per update
```

#### Live Project Specific Features
```
- percent_funded: Current % of funding goal reached
- percent_time: % of campaign duration elapsed
- funding_ratio: percent_funded / percent_time
- projected_final_percent: Projected final funding %
- backers_per_day: Acquisition rate adjusted for elapsed time
```

#### Completed Project Features
```
- backers_per_day: Backer acquisition rate
- avg_pledge_amount: Average contribution per backer
- has_updates: Binary indicator of update presence
- update_frequency: Temporal distribution of updates
- likes_per_backer: Engagement level relative to backers
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
1. **Separate Feature Sets**: Different feature extraction for training vs. prediction
2. **Indirect Indicators**: Use engagement metrics rather than direct funding indicators for completed projects
3. **Cross-Validation**: Properly stratified validation splits
4. **Feature Filtering**: Explicit removal of leaky features like final funding percentage

### 3. Random Forest Implementation with Cross-Validation

The model uses scikit-learn's RandomForestClassifier with stratified k-fold cross-validation for reliable performance assessment.

#### Model Configuration
```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'  # Handles class imbalance
)
```

#### Cross-Validation Process
```python
# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
```

#### Multiple Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **F1 Score**: Balanced measure of precision and recall
- **ROC AUC**: Discrimination ability across thresholds
- **Confusion Matrix**: Detailed breakdown of predictions

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
Cross-validation accuracy: ~70-80% (varies by dataset)
- More reliable than single test split accuracy
- F1 score: ~75-85% (balanced precision/recall)
- ROC AUC: ~0.8-0.9 (good discrimination ability)
```

#### Top Influencing Factors (examples)
1. Backers per day (acquisition rate)
2. Update engagement metrics
3. Campaign duration
4. Update frequency and timing
5. Key words in updates

### 6. Usage

1. Install requirements:
```bash
pip3 install -r requirements.txt
```

2. Run the analysis:
```bash
python3 update_analysis.py
```

3. Check results in the generated `results_[timestamp]` directory:
- `analysis_summary.txt`: Detailed analysis report
- `feature_importance_readable.csv`: Feature rankings
- `feature_importance.png`: Visual representation of important features
- `confusion_matrix.png`: Model performance visualization
- `live_projects_status.png`: Visual plot of live project trajectories
- `projected_funding_distribution.png`: Distribution of projected outcomes
- `live_projects_predictions.csv`: Detailed predictions for all live projects

### 7. Advantages of this Implementation

1. **Holistic Project Analysis**
   - Handles both live and completed projects
   - Provides actionable insights for ongoing campaigns
   - Realistic performance metrics through cross-validation

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

### 8. Limitations and Future Improvements

1. **Model Enhancements**
   - Implement gradient boosting models (XGBoost, LightGBM)
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