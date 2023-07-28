import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file
data = pd.read_csv('er2013_mini_XCEPTION.csv')

# Extract image pixels and labels
pixels = data['pixels'].tolist()
labels = data['emotion'].tolist()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(pixels, labels, test_size=0.2, random_state=42)
