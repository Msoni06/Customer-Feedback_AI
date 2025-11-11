import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Feedback AI",
    layout="wide"
)

@st.cache_resource
def load_models():
    print("Loading models...")
    try:
        sentiment_pipeline = pipeline("text-classification", model="./sentiment_model")
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        st.error("Please make sure the './sentiment_model' directory exists and was trained successfully.")
        return None, None
        
    try:
        summarizer_pipeline = pipeline("summarization", model="t5-small")
    except Exception as e:
        st.error(f"Error loading summarizer model: {e}")
        return None, None
        
    print("Models loaded.")
    return sentiment_pipeline, summarizer_pipeline

sentiment_pipeline, summarizer_pipeline = load_models()

@st.cache_data
def get_nltk_data():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

stop_words, lemmatizer = get_nltk_data()

@st.cache_data
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(cleaned_tokens)

@st.cache_data
def create_wordcloud(text):
    if not text.strip():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

st.title("Intelligent Customer Feedback Analysis System using AI")

if sentiment_pipeline is None or summarizer_pipeline is None:
    st.stop()

tab1, tab2 = st.tabs(["Single Feedback Analysis", "Batch Insights & Trends"])

with tab1:
    st.header("Analyze a Single Piece of Feedback")
    text_input = st.text_area("Enter customer feedback here:", height=150,
                              placeholder="e.g., 'The app is great, but customer service was slow.'")
    
    if st.button("Analyze Feedback", type="primary"):
        if text_input:
            with st.spinner("Analyzing..."):
                
                sentiment_result = sentiment_pipeline(text_input)[0]
                sentiment_label = sentiment_result['label']
                sentiment_score = sentiment_result['score']
                
                if sentiment_label == "Positive":
                    st.success(f"Sentiment: {sentiment_label} (Score: {sentiment_score:.2f})")
                elif sentiment_label == "Negative":
                    st.error(f"Sentiment: {sentiment_label} (Score: {sentiment_score:.2f})")
                else:
                    st.warning(f"Sentiment: {sentiment_label} (Score: {sentiment_score:.2f})")

                st.subheader("Summarization")
                summary = summarizer_pipeline(text_input, max_length=100, min_length=10)[0]['summary_text']
                st.info(summary)
        else:
            st.warning("Please enter some feedback to analyze.")

with tab2:
    st.header("Upload Feedback Data (CSV)")
    uploaded_file = st.file_uploader("Upload your cleaned_feedback.csv file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
            
        st.dataframe(df.head(), use_container_width=True)
        st.info(f"Loaded {len(df)} records.")
        
        if 'feedback' not in df.columns or 'date' not in df.columns:
            st.error("Error: CSV must contain 'feedback' and 'date' columns.")
            st.stop()

        if st.button("Generate Dashboard", type="primary"):
            with st.spinner("Generating insights... This may take a moment."):
                
                feedback_list = df['feedback'].dropna().astype(str).tolist()
                try:
                    sentiments = sentiment_pipeline(feedback_list, batch_size=8)
                    df['sentiment_label'] = [s['label'] for s in sentiments]
                except Exception as e:
                    st.warning(f"Batch processing failed ({e}), switching to row-by-row (slower).")
                    df['sentiment_label'] = df['feedback'].apply(lambda x: sentiment_pipeline(str(x))[0]['label'] if pd.notna(x) else None)
                
                df['preprocessed_text'] = df['feedback'].dropna().apply(preprocess_text)
                
                score_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
                df['satisfaction_score'] = df['sentiment_label'].map(score_map)

                st.subheader("Key Metrics")
                total_feedback = len(df)
                total_negative = df[df['sentiment_label'] == 'Negative'].shape[0]
                avg_score = df['satisfaction_score'].mean()

                k_col1, k_col2, k_col3 = st.columns(3)
                k_col1.metric("Total Feedback", total_feedback)
                k_col2.metric("Total Negative Feedback", total_negative)
                k_col3.metric("Avg. Satisfaction Score", f"{avg_score:.2f} / 1.0", delta_color="off")
                
                st.divider()

                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Distribution")
                    sentiment_counts = df['sentiment_label'].value_counts().reset_index()
                    sentiment_counts.columns = ['sentiment_label', 'count']
                    fig_pie = px.pie(sentiment_counts, names='sentiment_label', values='count', 
                                     color='sentiment_label',
                                     color_discrete_map={'Positive':'green', 'Negative':'red', 'Neutral':'orange'})
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    st.subheader("Top Recurring Issues")
                    negative_feedback = df[df['sentiment_label'] == 'Negative']['preprocessed_text']
                    if not negative_feedback.empty:
                        negative_feedback_processed = negative_feedback.dropna()
                        vectorizer = TfidfVectorizer(ngram_range=(2, 3), max_features=10, stop_words='english')
                        tfidf_matrix = vectorizer.fit_transform(negative_feedback_processed)
                        feature_names = vectorizer.get_feature_names_out()
                        scores = tfidf_matrix.sum(axis=0).A1
                        
                        issue_scores = pd.DataFrame({'Issue': feature_names, 'Score': scores})
                        issue_scores = issue_scores.sort_values(by='Score', ascending=False)
                        
                        st.dataframe(issue_scores, use_container_width=True)
                    else:
                        st.info("No negative feedback found to analyze.")
                
                st.divider()

                st.subheader("Common Feedback Themes")
                wc_col1, wc_col2 = st.columns(2)
                with wc_col1:
                    st.text("Positive Feedback Word Cloud")
                    pos_text = " ".join(df[df['sentiment_label'] == 'Positive']['preprocessed_text'].dropna())
                    fig = create_wordcloud(pos_text)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info("No positive feedback to show.")
                with wc_col2:
                    st.text("Negative Feedback Word Cloud")
                    neg_text = " ".join(df[df['sentiment_label'] == 'Negative']['preprocessed_text'].dropna())
                    fig = create_wordcloud(neg_text)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info("No negative feedback to show.")

                st.divider()
                
                st.subheader("Sentiment Trend Over Time")
                df['date'] = pd.to_datetime(df['date'])
                df_trend = df.set_index('date').resample('W')['satisfaction_score'].mean().reset_index()
                df_trend.columns = ['Week', 'Average Satisfaction Score']
                
                fig_line = px.line(df_trend, x='Week', y='Average Satisfaction Score', 
                                   title="Weekly Average Satisfaction Score")
                fig_line.update_traces(line_color='blue')
                st.plotly_chart(fig_line, use_container_width=True)