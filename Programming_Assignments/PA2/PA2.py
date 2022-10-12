import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#movies = pd.read_csv('movie-lens-data/movies.csv')
#links = pd.read_csv('movie-lens-data/links.csv')
ratings = pd.read_csv('movie-lens-data/ratings.csv')
#tags = pd.read_csv('movie-lens-data/tags.csv')

# Matrix containing all ratings per user per movie
'''movie_matrix = ratings.pivot_table(index = 'movieId', columns = 'userId', values = 'rating') # shape = (9066, 671)
# Normalize matrix
matrix_norm = movie_matrix.subtract(movie_matrix.mean(axis = 0), axis = 1) # shape = (9066, 671) (row, col)
# Cosine Similarity (movie to movie)
corr_matrix = pd.DataFrame(cosine_similarity(matrix_norm.fillna(0))) # shape = (9066, 9066)
# Set correlation matrix columns and indexes as 'movieId'
movieIds = pd.Series(movie_matrix.index)
corr_matrix.set_index([movieIds.values], inplace=True)
corr_matrix.columns = [movieIds.values]'''

movie_matrix = ratings.pivot_table(index = 'userId', columns = 'movieId', values = 'rating') # shape = (9066, 671)
corr2_matrix = movie_matrix.corr(method = 'pearson')

# Looping through each user (i)
for user in range(1,10): #len(movie_matrix)
    user_ratings = movie_matrix.iloc[user - 1].dropna() # gives us row for ith user
    recommend = pd.Series([],dtype=pd.StringDtype()) 
    temp = []
    
    # Looping through each movie for each user
    for movie in range(0, len(user_ratings)):
        similar = pd.Series(corr2_matrix[user_ratings.index[movie]]).dropna() 
        similar = similar.map(lambda x: x * user_ratings.values[movie])
        temp.append(similar)
        recommend = pd.concat(temp)

    recommend.sort_values(inplace = True, ascending = False)
    x = pd.DataFrame(recommend)
    recommend_filter = x[~x.index.isin(user_ratings.index)]
    #print(user, recommend.head(5))
    print(user, recommend_filter.index[0], 
          recommend_filter.index[1],
          recommend_filter.index[2],
          recommend_filter.index[3],
          recommend_filter.index[4])