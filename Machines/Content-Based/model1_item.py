import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# 데이터셋 로드 (예시 경로, 실제 경로에 맞게 수정해야 함)
item_data = pd.read_csv('./final_ITEM_DATA1.csv')

# 장르 데이터를 하나의 문자열로 결합
genre_columns = ['Action', 'Adventure', 'Casual', 'Indie', 'RPG', 'Simulation', 'Strategy', 'Sports', 'Racing']
item_data['Combined_Genres'] = item_data[genre_columns].apply(lambda row: ' '.join(row.index[row == 1]), axis=1)

# TF-IDF 벡터라이저를 사용하여 장르 특성을 벡터화
vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(item_data['Combined_Genres'])

# 게임 이름과 인덱스 매핑 (추천 시 사용)
game_indices = pd.Series(item_data.index, index=item_data['Name']).drop_duplicates()

# 특정 게임에 대한 유사도만 계산하는 함수
def recommend_games_by_genre(game_name, genre_matrix=genre_matrix, game_indices=game_indices, top_n=5):
    idx = game_indices[game_name]
    cosine_similarities = linear_kernel(genre_matrix[idx], genre_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n - 1:-1][::-1]
    return item_data['Name'].iloc[similar_indices].tolist()

# 예시 게임 이름을 기반으로 추천
game_example = 'Steam Squad'
recommended_games = recommend_games_by_genre(game_example)

print(f"'{game_example}' 장르와 유사한 게임 추천 목록: {recommended_games}")