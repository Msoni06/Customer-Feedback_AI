import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from prophet import Prophet
import matplotlib.pyplot as plt

print("Loading cleaned data for insights...")
try:
    df = pd.read_csv("cleaned_feedback.csv")
except FileNotFoundError:
    print("Error: cleaned_feedback.csv not found.")
    print("Please run data_preprocessing.py first.")
    exit()

print("\n--- Top 10 Recurring Issues (from Negative Feedback) ---")

negative_df = df[df['sentiment'] == 'Negative']['preprocessed_feedback'].dropna()

if not negative_df.empty:
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), max_features=20, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(negative_df)

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    issue_scores = pd.DataFrame({'issue': feature_names, 'score': scores})

    top_issues = issue_scores.sort_values(by='score', ascending=False).head(10)
    print(top_issues.to_string(index=False))
else:
    print("No negative feedback found to analyze.")

print("\n--- Generating Satisfaction Score Forecast ---")

df['date'] = pd.to_datetime(df['date'])

score_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
df['satisfaction_score'] = df['sentiment'].map(score_map)

df_daily = df.set_index('date').resample('D')['satisfaction_score'].mean().reset_index()
df_daily = df_daily.dropna()

df_prophet = df_daily.rename(columns={"date": "ds", "satisfaction_score": "y"})

if len(df_prophet) > 2:
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    print("Forecast generated. Saving plots...")

    fig1 = m.plot(forecast)
    plt.title("Satisfaction Score Forecast (Next 30 Days)")
    plt.xlabel("Date")
    plt.ylabel("Avg. Satisfaction Score")
    fig1.savefig("forecast_plot.png")

    fig2 = m.plot_components(forecast)
    fig2.savefig("forecast_components.png")

    print("Saved 'forecast_plot.png' and 'forecast_components.png'.")
    print("Combine these plots and the top issues into your 'AI_insights_report.pdf'.")
else:
    print("Not enough daily data to generate a forecast.")