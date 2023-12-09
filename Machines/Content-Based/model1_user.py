import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

user_data = pd.read_csv('./user_data.csv')

# 사용자별 게임 플레이 시간을 기반으로 사용자 프로필 생성
user_profiles = user_data.pivot_table(index='userId', columns='gameName', values='playTime', fill_value=0)

# 게임 간 유사도 계산
game_similarity_matrix = cosine_similarity(user_profiles.T)

# 유사도 데이터프레임 생성
game_similarity_df = pd.DataFrame(game_similarity_matrix, index=user_profiles.columns, columns=user_profiles.columns)

# 사용자에게 게임 추천하는 함수
def recommend_games_content_based(user_id, user_profiles, game_similarity_df, top_n=5):
    user_played_games = user_profiles.loc[user_id]
    user_played_games = user_played_games[user_played_games > 0]

    game_scores = pd.Series(dtype=float)
    for game in user_played_games.index:
        similarity_score = game_similarity_df[game]
        game_scores = game_scores.add(similarity_score, fill_value=0)

    game_scores = game_scores.drop(user_played_games.index)
    game_recommendations = game_scores.nlargest(top_n).index.tolist()

    return game_recommendations

# 사용자 ID를 지정하여 게임 추천 (예시 ID, 실제 ID로 변경 필요)
user_id_example = 151603712
recommended_games = recommend_games_content_based(user_id_example, user_profiles, game_similarity_df)

print(f"[{user_id_example}]에게 추천할 게임 목록: {recommended_games}")
