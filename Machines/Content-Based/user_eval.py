import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 데이터셋 불러오기
user_data = pd.read_csv('./user_data.csv')

# 사용자 프로필 생성
user_profiles = user_data.pivot_table(index='userId', columns='gameName', values='playTime', fill_value=0)

# 게임 간 피어슨 상관계수 계산
game_similarity_matrix = user_profiles.corr()

# 유사도 데이터프레임 생성
game_similarity_df = pd.DataFrame(game_similarity_matrix, index=user_profiles.columns, columns=user_profiles.columns)

# 사용자에게 게임 추천하는 함수
def recommend_games_content_based(user_id, user_profiles, game_similarity_df, top_n=10):
    user_played_games = user_profiles.loc[user_id]
    user_played_games = user_played_games[user_played_games > 0]

    game_scores = pd.Series(dtype=float)
    for game in user_played_games.index:
        similarity_score = game_similarity_df[game]
        game_scores = game_scores.add(similarity_score, fill_value=0)

    game_scores = game_scores.drop(user_played_games.index)
    game_recommendations = game_scores.nlargest(top_n).index.tolist()

    return game_recommendations

# 사용자 ID를 지정하여 게임 추천
user_id_example = 151603712
recommended_games = recommend_games_content_based(user_id_example, user_profiles, game_similarity_df)

# 추천된 게임의 예상 플레이 시간 (여기서는 간단하게 평균 플레이 시간을 사용)
predicted_playtimes = user_profiles[recommended_games].mean()

# 실제 사용자가 평가한 게임의 플레이 시간
actual_playtimes = user_profiles.loc[user_id_example]

# MSE와 MAE 계산 (예측된 게임과 실제 게임 간의 매칭을 고려하여 계산)
common_games = actual_playtimes.index.intersection(predicted_playtimes.index)
mse = mean_squared_error(actual_playtimes[common_games], predicted_playtimes[common_games])
mae = mean_absolute_error(actual_playtimes[common_games], predicted_playtimes[common_games])

print("추천된 게임:", recommended_games)
print("MSE:", mse)
print("MAE:", mae)
