import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

print("Generating simulated data...")

positive_templates = [
    "The customer service was excellent!", "I love this product, it works perfectly.",
    "Very satisfied with the quick delivery.", "Great experience, will buy again.",
    "The app is so easy to use and very fast."
]
negative_templates = [
    "The website is #awful and slow.", "I am very unhappy with the support team.",
    "My order arrived late and was damaged.", "Product broke after one use. ##TERRIBLE",
    "Can't find what I'm looking for, navigation is confusing."
]
neutral_templates = [
    "The product is okay, it does the job.", "Shipping time was average.",
    "I haven't used the support team yet.", "The price is fair for what you get.",
    "The new update is fine, no major changes."
]

data = []
start_date = datetime.now() - timedelta(days=365)

for i in range(1200):  # Generate 1200 records
    date = start_date + timedelta(days=random.randint(0, 364), hours=random.randint(0, 23))

    if i % 10 == 0:  # 10% neutral
        text = random.choice(neutral_templates)
        sentiment = "Neutral"
    elif i % 3 == 0:  # ~30% negative
        text = random.choice(negative_templates)
        sentiment = "Negative"
    else:  # ~60% positive
        text = random.choice(positive_templates)
        sentiment = "Positive"

    if random.random() < 0.05:  # 5% chance of being a duplicate
        if data: text = data[-1]['feedback']
    if random.random() < 0.05:  # 5% chance of being NaN
        text = np.nan

    data.append({
        "feedback_id": f"fb_{i:04d}",
        "date": date,
        "feedback": text,
        "sentiment": sentiment
    })

df = pd.DataFrame(data)
df.to_csv("simulated_feedback.csv", index=False)

print("Saved simulated_feedback.csv")