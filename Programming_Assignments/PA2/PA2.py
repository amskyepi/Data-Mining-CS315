import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Read in ratings.csv
ratings = pd.read_csv('movie-lens-data/ratings.csv')

# Matrix containing all ratings per user per movie
movie_matrix = ratings.pivot_table(index = 'movieId', columns = 'userId', values = 'rating') # shape = (9066, 671)

# Normalize matrix
matrix_norm = movie_matrix.subtract(movie_matrix.mean(axis = 0), axis = 1) # shape = (9066, 671) (row, col)

# Cosine Similarity (movie to movie)
corr_matrix = pd.DataFrame(cosine_similarity(matrix_norm.fillna(0))) # shape = (9066, 9066)

# Set correlation matrix columns and indices as 'movieId'
movieIds = pd.Series(movie_matrix.index)
corr_matrix.set_index(movieIds.values, inplace=True)
corr_matrix.columns = movieIds.values
movie_matrix = ratings.pivot_table(index = 'userId', columns = 'movieId', values = 'rating') # shape = (9066, 671)

# Looping through each user (i)
for user in range(1, len(movie_matrix) + 1):
    user_ratings = movie_matrix.iloc[user - 1].dropna() # gives us row for ith user
    recommend = pd.Series([],dtype=pd.StringDtype()) 
    temp = []
    
    # Looping through each movie for each user
    for movie in range(0, len(user_ratings)):
        # Selects given movie data from correlation matrix
        # (this creates a list named 'similar' to show how the
        # user might rate each movie in movie_matrix based on their rating
        # of the movie we are observing in corr_matrix)
        similar = pd.Series(corr_matrix[user_ratings.index[movie]]).dropna()
        
        # Predicts user rating of each movie in list 
        similar = similar.map(lambda x: x * user_ratings.values[movie])
        temp.append(similar)
        recommend = pd.concat(temp)

    # Sort movie recommendations
    recommend.sort_values(inplace = True, ascending = False)
    x = pd.DataFrame(recommend)
    
    # Filter out movies that have not been rated by user
    recommend_filter = x[~x.index.isin(user_ratings.index)]

    # Write results (recommendations) to output file
    with open('output.txt', 'a') as output_file:
        output_file.write(str(user) + ' ')
        for i in range(0,5):
            line = (str(recommend_filter.index[i]) + ' ')
            output_file.write(line)
        output_file.write('\n')
        output_file.close()