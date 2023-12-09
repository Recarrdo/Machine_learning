#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean


# In[13]:


# Load game data
data = pd.read_csv("./final_ITEM_DATA1.csv")

# Load user data
userData = pd.read_csv("./user_data.csv")

# Get unique gameName values from another CSV file
valid_game_names = userData['gameName'].unique()

# Filter rows in 'data' dataframe to keep only those with gameName in valid_game_names
data = data[data['Name'].isin(valid_game_names)]

print(len(data))


# In[14]:


# Select features like genre and positive, negative, price
features = [
            'Positive', 'Negative', 'Recommendations', 'Peak CCU', 'Estimated owners', 'Price', 
            'Action', 'Adventure', 'Animation & Modeling', 'Audio Production', 'Casual', 
            'Design & Illustration', 'Documentary', 'Early Access', 'Education', 'Episodic', 'Free to Play', 
            'Game Development', 'Gore', 'Indie', 'Massively Multiplayer', 'Movie', 'Nudity', 'Photo Editing', 'RPG', 
            'Racing', 'Sexual Content', 'Short', 'Simulation', 'Software Training', 'Sports', 'Strategy', 
            'Tutorial', 'Utilities', 'Video Production', 'Violent', 'Web Publishing'
            ]


# In[15]:


# Select necessary columns
selected_features = data[features]

# Data normalization
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(selected_features)

# Reduce data to 2 dimensions using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

#boto3.Session.resource('s3').Bucket(bucket).Object(key).upload_file()


# In[16]:


# Apply K-means clustering
kmeans = KMeans(n_clusters=15, n_init=10, random_state=75)
kmeans.fit(pca_result)

# data_location = 's3://{}'.format(bucket)

# kmeans = KMeans(
#                 role = role,
#                 train_instance_count = 1,
#                 train_instance_type = 'ml.t3.medium',
#                 k=10,
#                 data_location = data_location)


# In[17]:


# Check clustering results
data['Cluster_Labels'] = kmeans.labels_


# In[18]:


# Visualization of results
plt.figure(figsize=(8, 6))
for cluster_label in range(10):
    cluster_data = pca_result[kmeans.labels_ == cluster_label]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_label}')

plt.title('K-means Clustering of Steam games')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# In[19]:


# Calculate silhouette score
silhouette_avg = silhouette_score(pca_result, kmeans.labels_)
print(f"\nSilhouette Score: {silhouette_avg}\n")


# In[20]:


def calculate_weighted_average(user_games_info, data, features):
    weighted_averages = {}
    
    for feature in features:
        feature_weighted_sum = 0

        for idx, game in user_games_info.iterrows():
            game_data = data[data['Name'] == game['gameName']][feature]
            
            # Process only if game data exists
            if not game_data.empty:
                game_weight = game['percentage_playTime']
                feature_weighted_sum += (game_data.values[0] * game_weight)
                

        weighted_averages[feature] = feature_weighted_sum

    return pd.DataFrame(weighted_averages, index=[0])


# In[21]:


# User ID of the selected user in userData
user_id = 53875128


# In[22]:


# Get information on all games played by the selected user (only game name and play time)
user_games_info = userData[userData['userId'] == user_id][['gameName', 'playTime']]

# List of names of all games played by the selected user
user_games = user_games_info['gameName'].tolist()

# Get information on all games
user_game_info = data[data['Name'].isin(user_games)]

# Scale playtime to a percentage between 0 and 1
sum_play_time = user_games_info['playTime'].sum()
user_games_info['percentage_playTime'] = user_games_info['playTime'] / sum_play_time


# In[23]:


# Calculate weighted average for each feature
user_game_average = calculate_weighted_average(user_games_info, data, features)
print("[User's average game data]")
print(user_game_average)

# set test data to average game data of user
test_data = user_game_average.copy()


# In[24]:


# Data normalization
scaled_test_data = scaler.transform(test_data)
pca_test_result = pca.transform(scaled_test_data)

# Predict the cluster of new data
test_cluster = kmeans.predict(pca_test_result)[0]


# In[25]:


# Extract data points from the selected cluster
cluster_indices = np.where(kmeans.labels_ == test_cluster)[0]
cluster_data_points = pca_result[cluster_indices]

# Calculate Spearman correlation coefficients and distances for each data point in the cluster
spearman_distances = []
for idx, point in enumerate(cluster_data_points):
    if np.array_equal(point, pca_test_result[0]):
        continue  # Skip the test data point itself
    
    # Calculate Spearman correlation coefficient
    spearman_coeff, _ = spearmanr(selected_features.iloc[cluster_indices[idx]], test_data.values[0])
    
    # Calculate Euclidean distance between points
    euclidean_dist = euclidean(point, pca_test_result[0])
    
    # Append tuple containing index, Spearman coefficient, and Euclidean distance
    spearman_distances.append((cluster_indices[idx], spearman_coeff, euclidean_dist))

# Sort by Spearman coefficient in descending order
spearman_distances.sort(key=lambda x: x[1], reverse=True)

# Display the top 30 similar games
top_similar_games = spearman_distances[:20]
print("Top 20 similar games based on Spearman correlation coefficient:")
for i, (index, spearman_coeff, euclidean_dist) in enumerate(top_similar_games, 1):
    game_name = data.iloc[index]['Name']
    print(f"{i}. Game: {game_name}, Spearman Coefficient: {spearman_coeff}, Euclidean Distance: {euclidean_dist}")
    
print()
    

# Sort by Euclidean distance in ascending order
spearman_distances.sort(key=lambda x: x[2])

# Display the top 30 similar games based on Euclidean distance
top_similar_games_euclidean = spearman_distances[:20]
print("Top 20 similar games based on Euclidean distance:")
for i, (index, spearman_coeff, euclidean_dist) in enumerate(top_similar_games_euclidean, 1):
    game_name = data.iloc[index]['Name']
    print(f"{i}. Game: {game_name}, Spearman Coefficient: {spearman_coeff}, Euclidean Distance: {euclidean_dist}")


# In[26]:


def calculate_accuracy_precision_recall(user_games, top_similar_games):
    # Get the names of recommended games
    recommended_games = [data.iloc[index]['Name'] for index, _, _ in top_similar_games]

    # Calculate accuracy, precision, and recall
    total_recommended = len(recommended_games)
    correctly_recommended = len(set(user_games).intersection(recommended_games))
    total_user_games = len(user_games)

    accuracy = correctly_recommended / total_recommended if total_recommended != 0 else 0
    recall = correctly_recommended / total_user_games if total_user_games != 0 else 0

    return accuracy, recall


# In[27]:


# User games list (games actually played by the user)
user_games_list = user_games_info['gameName'].tolist()

# Calculate accuracy, precision, and recall
accuracy, recall = calculate_accuracy_precision_recall(user_games_list, top_similar_games)

# Display accuracy, precision, and recall
print("[Evaluation - Spearman]")
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print()

# Calculate accuracy, precision, and recall
accuracy_e, recall_e = calculate_accuracy_precision_recall(user_games_list, top_similar_games_euclidean)

# Display accuracy, precision, and recall
print("[Evaluation - Euclidean]")
print(f"Accuracy: {accuracy_e}")
print(f"Recall: {recall_e}")


# In[28]:


print(user_games_info['gameName'])


# In[29]:


# Get the indices of played recommended games from Spearman coefficient based recommendation
played_recommended_games_indices = [index for index, _, _ in top_similar_games if data.iloc[index]['Name'] in user_games]

if played_recommended_games_indices:
    print("User played the following games from Spearman coefficient based recommendations and their rankings:")
    for idx, index in enumerate(played_recommended_games_indices, 1):
        ranking = next(i+1 for i, (idx_, _, _) in enumerate(top_similar_games) if idx_ == index)
        print(f"{idx}. Game: {data.iloc[index]['Name']}, Ranking: {ranking}")
else:
    print("The user didn't play any of the Spearman coefficient based recommended games.")

# Get the indices of played recommended games from Euclidean distance based recommendation
played_recommended_games_indices_euclidean = [index for index, _, _ in top_similar_games_euclidean if data.iloc[index]['Name'] in user_games]

if played_recommended_games_indices_euclidean:
    print("\nUser played the following games from Euclidean distance based recommendations and their rankings:")
    for idx, index in enumerate(played_recommended_games_indices_euclidean, 1):
        ranking = next(i+1 for i, (idx_, _, _) in enumerate(top_similar_games_euclidean) if idx_ == index)
        print(f"{idx}. Game: {data.iloc[index]['Name']}, Ranking: {ranking}")
else:
    print("\nThe user didn't play any of the Euclidean distance based recommended games.")






