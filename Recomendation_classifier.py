# Data processing
import pandas as pd
import numpy as np
import scipy.stats

# Visualization
import seaborn as sns
from matplotlib import pyplot as plt

# Similarity
from sklearn.metrics.pairwise import cosine_similarity


"""
------------------------------------------------
 merged.csv info:
------------------------------------------------
 0   user_id          100836 non-null  int64  
 1   movie_id         100836 non-null  int64  
 2   rating           100836 non-null  float64
 3   title            100836 non-null  object 
 4   genres           100836 non-null  object 
 5   is_adult         100762 non-null  float64
 6   runtime_min      100761 non-null  float64
 7   avgRating_IMDB   100762 non-null  float64
 8   numVotes_IMDB    100762 non-null  float64
 9   avgRating_users  0 non-null       float64
 10  numVotes_users   0 non-null       float64
 11  timestamp        100836 non-null  int64  
 12  IMDB_id          100836 non-null  int64  
------------------------------------------------
"""


def load_data(filepath):
    ratings = pd.read_csv(filepath)
    print(ratings.head(), "\n")
    print('The ratings dataset has', ratings['user_id'].nunique(), 'unique users', "\n")
    print('The ratings dataset has', ratings['movie_id'].nunique(), 'unique movies', "\n")
    print('The ratings dataset has', ratings['rating'].nunique(), 'unique ratings', "\n")
    print('The unique ratings are', sorted(ratings['rating'].unique()), "\n")
    return ratings    

def preprocess_ratings(ratings):
    agg_ratings = ratings.groupby('title').agg(mean_rating=('rating', 'mean'),
                                      number_of_ratings=('rating', 'count')).reset_index()
    
    agg_ratings_GT100 = agg_ratings[agg_ratings['number_of_ratings'] > 100]

    visualize_ratings(agg_ratings_GT100)

    df_GT100 = pd.merge(ratings, agg_ratings_GT100[['title']], on='title', how='inner')
   
    return df_GT100
    
def visualize_ratings(agg_ratings_GT100):
    sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_GT100)
    plt.show()    

def create_user_item_matrix(df_GT100):
    matrix = df_GT100.pivot_table(index='title', columns='user_id', values='rating')
    matrix_norm = matrix.subtract(matrix.mean(axis=1), axis=0)
    return matrix_norm    

def calculate_item_similarity(matrix_norm):
    item_similarity = matrix_norm.T.corr()
    return item_similarity

def predict_rating(matrix_norm, item_similarity, picked_userid, picked_movie):
    picked_userid_watched = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all') \
                                    .sort_values(ascending=False)) \
                                    .reset_index() \
                                    .rename(columns={1: 'rating'})
    
    picked_movie_similarity_score = item_similarity[[picked_movie]].reset_index() \
        .rename(columns={'American Pie (1999)': 'similarity_score'})
    
    picked_userid_watched_similarity = pd.merge(left=picked_userid_watched,
                                            right=picked_movie_similarity_score,
                                            on='title',
                                            how='inner') \
                                       .sort_values('similarity_score', ascending=False)[:5]
    
    predicted_rating = round(np.average(picked_userid_watched_similarity['rating'],
                                    weights=picked_userid_watched_similarity['similarity_score']), 6)
    return predicted_rating

def item_based_rec(matrix_norm, item_similarity, picked_userid, number_of_similar_items, number_of_recommendations):
    picked_userid_unwatched = pd.DataFrame(matrix_norm[picked_userid].isna()).reset_index()
    picked_userid_unwatched = picked_userid_unwatched[picked_userid_unwatched[1] == True]['title'].values.tolist()
    picked_userid_watched = pd.DataFrame(matrix_norm[picked_userid].dropna(axis=0, how='all')
                                         .sort_values(ascending=False)).reset_index().rename(columns={1: 'rating'})
    
    rating_prediction = {}
    
    for picked_movie in picked_userid_unwatched:
        picked_movie_similarity_score = item_similarity[[picked_movie]]\
            .reset_index().rename(columns={picked_movie:'similarity_score'})
        picked_userid_watched_similarity = pd.merge(left=picked_userid_watched,
                                                    right=picked_movie_similarity_score,
                                                    on='title', how='inner')\
                                                .sort_values('similarity_score', ascending=False)[:number_of_similar_items]
        predicted_rating = round(np.average(picked_userid_watched_similarity['rating'],
                                            weights=picked_userid_watched_similarity['similarity_score']), 6)
        rating_prediction[picked_movie] = predicted_rating
    return sorted(rating_prediction.items(), key=lambda x: x[1], reverse=True)[:number_of_recommendations]

def main():
    # Usage example:
    filepath = 'merged.csv'
    ratings = load_data(filepath)
    agg_ratings_GT100 = preprocess_ratings(ratings)
    matrix_norm = create_user_item_matrix(agg_ratings_GT100)
    item_similarity = calculate_item_similarity(matrix_norm)


    predicted_rating = predict_rating(matrix_norm, item_similarity, picked_userid=1, picked_movie='American Pie (1999)')
    print(f'The predicted rating for American Pie (1999) by user 1 is {predicted_rating}\n')

    recommended_movies = item_based_rec(matrix_norm, item_similarity, picked_userid=1, number_of_similar_items=5, number_of_recommendations=3)
    print(f'The top 3 recommended movies for user 1 are {recommended_movies}\n')
    print("Done\n")

if __name__ == "__main__":
    main()
