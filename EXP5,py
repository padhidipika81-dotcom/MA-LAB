# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Create dataset
data = {
    'weather': ['sunny', 'sunny', 'rainy', 'rainy'],
    'play_game': ['yes', 'yes', 'no', 'no']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical data
le_weather = LabelEncoder()
le_play = LabelEncoder()

df['weather'] = le_weather.fit_transform(df['weather'])
df['play_game'] = le_play.fit_transform(df['play_game'])

# Split features and target
X = df[['weather']]
y = df['play_game']

# Train Naive Bayes model
model = GaussianNB()
model.fit(X, y)

# Predict for 'sunny'
test = le_weather.transform(['sunny']).reshape(-1, 1)
prediction = model.predict(test)

# Convert prediction back to label
result = le_play.inverse_transform(prediction)

print("Prediction for weather = sunny:", result[0])
