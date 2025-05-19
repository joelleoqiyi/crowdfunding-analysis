import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeClassifier
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from sklearn.tree import export_graphviz
from sklearn.feature_extraction.text import CountVectorizer
import graphviz
import re

# === UI: Tailwind-consistent styling ===
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC; /* bg-slate-50 */
        color: #1E293B; /* text-slate-800 */
    }

    .stApp {
        padding: 2rem;
    }

    h1, h2, h3, h4 {
        font-weight: 600;
        color: #1E293B;
    }

    .stTextInput input,
    .stTextArea textarea,
    .stSlider,
    .stCheckbox,
    .stSelectbox div {
        border-radius: 0.5rem;
        border: 1px solid #CBD5E1; /* border-slate-300 */
        padding: 0.5rem;
    }

    .stAlert {
        border-radius: 0.75rem !important;
        padding: 1rem !important;
        font-size: 0.95rem;
    }

    .st-success {
        background-color: #ECFDF5 !important; /* bg-emerald-50 */
        border-left: 5px solid #059669 !important; /* text-emerald-600 */
    }

    .st-warning {
        background-color: #FFFBEB !important; /* bg-yellow-50 */
        border-left: 5px solid #CA8A04 !important; /* text-yellow-600 */
    }

    .stButton button {
        background-color: #059669; /* bg-emerald-600 */
        color: white;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border-radius: 0.5rem;
        border: none;
    }

    .stButton button:hover {
        background-color: #047857; /* bg-emerald-700 */
    }

    .stMarkdown h3 {
        font-size: 1.2rem;
        margin-top: 1.5rem;
        color: #1E293B;
    }

    .stExpanderHeader {
        font-size: 1rem;
        font-weight: 500;
        color: #1E293B;
    }
    </style>
""", unsafe_allow_html=True)

# === Load saved models ===
model = joblib.load("decision_tree_model.pkl")
vectorizer_dt = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")
vectorizer_tfidf = joblib.load("tfidf_vectorizer.pkl")
logistic_model = joblib.load("logistic_model.pkl")

# === Main Title ===
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='font-size: 2.5rem;'>Kickstarter Campaign Oracle</h1>
        <p style='font-size: 1.1rem; color: #6B7280;'>
            Leverage machine learning to predict your campaign’s success based on description, category, and engagement data.
        </p>
    </div>
""", unsafe_allow_html=True)

# === Sidebar ===
st.sidebar.header("Campaign Settings (Dials)")
campaign_name = st.sidebar.text_input("Campaign Name", "EcoSmart Charger")
category = st.sidebar.text_input("Category", "Gadgets")
campaign_summary = st.sidebar.text_area("Campaign Summary", "A portable, solar-powered charger with AI integration for outdoor enthusiasts.")
goal = st.sidebar.slider("Funding Goal (USD)", 1000, 100000, 58223)
backers = st.sidebar.slider("Number of Backers", 0, 2000, 1135)
fx_rate = st.sidebar.slider("FX Rate", 0.5, 2.0, 1.0)
staff_pick = st.sidebar.checkbox("Picked by Kickstarter staff", value=False)
staff_pick_val = 1 if staff_pick else 0

# === Preprocess input ===
combined_text = (category + " " + campaign_name + " " + campaign_summary).lower()
text_vector = vectorizer_dt.transform([combined_text])
numeric_array = np.array([[goal, fx_rate, backers, staff_pick_val]])
numeric_scaled = scaler.transform(numeric_array)
X_combined_input = hstack([text_vector, csr_matrix(numeric_scaled)])

# === Prediction ===
prediction = model.predict(X_combined_input)[0]
label = "Successful" if prediction == 1 else "Failed"
st.subheader(f"Prediction Result: **{label}**")

# === Suggestions and Feedback ===
success_signals = []
improvement_suggestions = []

if backers > 267:
    success_signals.append("High number of backers signals strong interest.")
else:
    improvement_suggestions.append("Increase backers — improve marketing or social reach.")

if goal <= 48720:
    success_signals.append("FX rate within expected range builds trust.")
else:
    improvement_suggestions.append("Reduce funding goal to a more realistic level.")

if fx_rate <= 1.306:
    success_signals.append("FX rate within expected range builds trust.")
else:
    improvement_suggestions.append("High FX rate may confuse backers. Consider adjusting pricing.")

if staff_pick:
    success_signals.append("Being staff-picked increases campaign credibility.")
else:
    improvement_suggestions.append("Consider improving campaign quality to get staff-picked status.")

# === Model-based keyword suggestions ===
X_text_input = vectorizer_tfidf.transform([combined_text])
feature_names = vectorizer_tfidf.get_feature_names_out()
coefs = logistic_model.coef_[0]
keyword_impact = dict(zip(feature_names, coefs))

# Whole-word match
def contains_whole_word(word, text):
    return re.search(rf'\b{re.escape(word)}\b', text) is not None

# Keyword impact
top_positive_keywords = sorted(keyword_impact.items(), key=lambda x: x[1], reverse=True)[:10]
top_negative_keywords = sorted(keyword_impact.items(), key=lambda x: x[1])[:10]

for word, weight in top_positive_keywords:
    if contains_whole_word(word, combined_text):
        success_signals.append(f"Using **'{word}'** helps! It's a strong success signal.")

for word, weight in top_negative_keywords:
    if contains_whole_word(word, combined_text):
        improvement_suggestions.append(f"Consider rephrasing or avoiding **'{word}'** — it often correlates with campaign failure.")

for word, weight in top_positive_keywords:
    if not contains_whole_word(word, combined_text):
        improvement_suggestions.append(f"Consider including keywords like **'{word}'**, which are strong success indicators.")

# === Display Results ===
with st.expander("What Your Campaign Does Well"):
    for tip in success_signals:
        st.success("- " + tip)

with st.expander("Suggestions to Improve"):
    for tip in improvement_suggestions:
        st.warning("- " + tip)

# === Visualise Decision Tree ===
df = pd.read_csv("./Kickstarter_TechFiltered_TechOnly.csv")
df = df.dropna(subset=['campaign_summary', 'state'])
df['success'] = df['state'].apply(lambda x: 1 if str(x).lower() == 'successful' else 0)
if 'campaign_name' not in df.columns:
    df['campaign_name'] = ''
df['combined_text'] = (
    df['category_name'].fillna('') + ' ' +
    df['campaign_name'].fillna('') + ' ' +
    df['campaign_summary'].fillna('')
).str.lower()

vectorizer = CountVectorizer(stop_words='english', max_features=30)
X_text = vectorizer.fit_transform(df['combined_text'])
y = df['success']
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_text, y)

st.markdown("### Visual Explanation of Decision Tree")
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=vectorizer.get_feature_names_out().tolist(),
    class_names=["Failed", "Successful"],
    filled=True,
    rounded=True,
    special_characters=True
)
st.graphviz_chart(dot_data)

with st.expander("View Full Decision Tree Notebook"):
    st.markdown("[Click here to view the full decision tree notebook](https://github.com/joelleoqiyi/crowdfunding-analysis/blob/Shi-Ying-Branch/eda_decisiontree.ipynb)", unsafe_allow_html=True)
