import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the data from the CSV file
file_path = './games.csv'  # Updated file path to access the uploaded file
data = pd.read_csv(file_path)

# Columns to be removed
columns_to_remove = [
    'Screenshots', 'Movies',  # URLs for screenshots and movies
    'About the game', 'Website', 'Support url', 'Support email',  # Descriptive text and additional URLs
    'Notes', 'Reviews', 'Metacritic url'  # Reviews, notes, and external links
]

# Removing the specified columns
data_reduced = data.drop(columns=columns_to_remove, errors='ignore')

# One-hot encoding for 'Genre' column
genre_df = data_reduced['Genres'].str.get_dummies(sep=',')
# Removing 'Genre' from data_reduced
data_reduced = data_reduced.drop('Genres', axis=1)

# Applying label encoding to remaining categorical columns, excluding 'Name'
categorical_cols = data_reduced.select_dtypes(include=['object']).columns
categorical_cols = categorical_cols.drop('Name')  # Exclude 'Name' from label encoding
for col in categorical_cols:
    label_encoder = LabelEncoder()
    data_reduced[col] = label_encoder.fit_transform(data_reduced[col].astype(str))

# Concatenating one-hot encoded 'Genre' with the rest of the data
data_reduced = pd.concat([data_reduced, genre_df], axis=1)

# Applying MinMaxScaler to numeric columns, excluding 'AppID'
numeric_cols = data_reduced.select_dtypes(include=['int64', 'float64']).columns
numeric_cols = numeric_cols.drop('AppID')  # Exclude 'AppID' from scaling
scaler = MinMaxScaler()
data_reduced[numeric_cols] = scaler.fit_transform(data_reduced[numeric_cols])

# Save the processed data to a new CSV file (optional)
processed_file_path = './processed_one2_games.csv'  # Updated file path for saving the processed data
data_reduced.to_csv(processed_file_path, index=False)

# Displaying the first few rows of the processed data
print(data_reduced.head())
