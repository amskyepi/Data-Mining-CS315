import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('movie-lens-data/movies.csv')
#links = pd.read_csv('movie-lens-data/links.csv')
ratings = pd.read_csv('movie-lens-data/ratings.csv')
#tags = pd.read_csv('movie-lens-data/tags.csv')

# Matrix containing all ratings per user per movie
movie_matrix = ratings.pivot_table(index = 'movieId', columns = 'userId', values = 'rating') # shape = (9066, 671)
# Normalize matrix
matrix_norm = movie_matrix.subtract(movie_matrix.mean(axis = 1), axis = 0) # shape = (9066, 671) (row, col)
# Cosine Similarity (movie to movie)
cos_matrix = cosine_similarity(matrix_norm.fillna(0)) # shape = (9066, 9066)
movie_matrix = ratings.pivot_table(index = 'userId', columns = 'movieId', values = 'rating') # shape = (9066, 671)

# List containing each movieId and corresponding index
movie_list = []
for x in movie_matrix.columns:
    movie_list.append(x)

# Looping through each user (i)
for i in range(1,2): #len(movie_matrix)
    user_ratings = movie_matrix.iloc[i - 1].dropna() # gives us row for ith user
    recommend = pd.Series([],dtype=pd.StringDtype()) 
    temp = []
    
    # Looping through each movie (j) for each user (i)
    for j in range(0, len(user_ratings)):
        # extract movie from similarity matrix
        # user_ratings[j] gives us the movieId of a rated movie
        print(user_ratings.index[j])
        print(cos_matrix[user_ratings.index[j]])
        similar = pd.Series(cos_matrix[user_ratings.index[j]]).dropna() 
        similar = similar.map(lambda x: x * user_ratings.values[j])
        temp.append(similar)
        recommend = pd.concat(temp)
        
    recommend.sort_values(inplace = True, ascending = False)
    x = pd.DataFrame(recommend)
    recommend_filter = x[~x.index.isin(user_ratings.index)]
    print(i, recommend_filter.index[0], 
          recommend_filter.index[1],
          recommend_filter.index[2],
          recommend_filter.index[3],
          recommend_filter.index[4])