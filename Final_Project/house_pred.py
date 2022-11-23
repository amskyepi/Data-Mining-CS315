import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error


# Reads data file and prepocesses data for training
def read_and_clean(file_name):
    # Open file
    df = pd.read_csv(file_name)
    
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Clean original sale amount
    df['ORIGINAL SALE AMOUNT'] = df['ORIGINAL SALE AMOUNT'].values.astype(str)
    df['ORIGINAL SALE AMOUNT'] = df['ORIGINAL SALE AMOUNT'].map(lambda x: str(x.lstrip('$'))).str.replace(',', '')
    
    # Categorical data -> numerical data
    df['STYLE'].replace(['1.5 FINISHED', 
                         '2 STORY', 
                         'RANCH', 
                         'BI-LEVEL (SPLIT ENTRY)', 
                         'SPLIT'], [1, 2, 3, 4, 5], inplace = True)
    
    # One Hot Encoding for categorical data
    area_list = ['YACOLT', 
                'AMBOY', 
                'LA CENTER', 
                'WOODLAND', 
                'RIDGEFIELD',
                'BATTLE GROUND',
                'BRUSH PRAIRIE',
                'VANCOUVER',
                'CAMAS',
                'WASHOUGAL']

    for item in area_list:
        df.loc[df['Parcel Address'].str.contains(item, na = False), 'Parcel Address'] = item    
    
    one_hot_encoder = OneHotEncoder()
    results = one_hot_encoder.fit_transform(df[['Parcel Address']])
    df = df.join(pd.DataFrame(results.toarray(), columns = one_hot_encoder.categories_))
    
    # Drop rows with empty values for specified columns
    df = df.dropna(axis = 0, subset = [' YEAR BUILT', 
                                       'STYLE',
                                       'QUALITY digital',
                                       'PARCEL SIZE ACRES',
                                       'MAIN AND UPPER LIVING AREA',])
    
    df['PID'] = df['PID'].values.astype(int)
    df[' YEAR BUILT'] = df[' YEAR BUILT'].values.astype(int)
    df['MAIN AND UPPER LIVING AREA'] = df['MAIN AND UPPER LIVING AREA'].values.astype(int)
    df['ORIGINAL SALE AMOUNT'] = df['ORIGINAL SALE AMOUNT'].values.astype(int)
    
    # Drop unwanted columns
    df = df.drop(columns = ['BUILDING TYPE',
                            'Parcel Address', 
                            'STYLE',
                            'QUALITY',
                            'BASEMENT AREA', 
                            'ADJUSTED SALE AMOUNT',
                            'PARCEL SIZE SQ FT',
                            'VIEW',
                            'WATERFRONT', 
                            'SALE DATE', 
                            'ASSESSOR NH (REFERENCE NO)'])
    
    df.insert(len(df.columns) - 1, 'ORIGINAL SALE AMOUNT', df.pop('ORIGINAL SALE AMOUNT'))
    
    # Do not include houses that are priced above the defined threshold
    upper_limit = 500000
    lower_limit = 0
    df = df[df['ORIGINAL SALE AMOUNT'] < upper_limit]
    df = df[df['ORIGINAL SALE AMOUNT'] > lower_limit]
    
    return(df)

# Returns unique values in dataframe column
def distinct_styles(df, column_name):
    return(df[column_name].unique())

# Splits training and test set
def train_test_splitting(df):
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Splitting dataset into training/testing
    print("Processing Data...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    
    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return(x_train, x_test, y_train, y_test)

# Decision Tree Algorithm
def decision_tree_learner(x_train, x_test, y_train, y_test):    
    # Training Decision Tree Classification model on training set
    print("Training...")
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(x_train, y_train)
    
    # Predicting test set results
    print("Testing...")
    y_pred = classifier.predict(x_test)
    
    return(y_pred, y_test)

# SVM Learning Algorithm
def svm_learner(x_train, x_test, y_train, y_test):
    # Training SVM model on training set
    print("Training...")
    model_SVR = svm.SVR()
    model_SVR.fit(x_train,y_train)
    
    # Predicting test set results
    print("Testing...")
    y_pred = model_SVR.predict(x_test)
    
    return(y_pred, y_test)

def linear_reg_learner(x_train, x_test, y_train, y_test):
    # Training Linear Regression model on training set
    print("Training...")
    model_LR = LinearRegression()
    model_LR.fit(x_train, y_train)
    
    # Predicting test set results
    print("Testing...")
    y_pred = model_LR.predict(x_test)
    
    return (y_pred, y_test)

def random_forest_learner(x_train, x_test, y_train, y_test):
    # Training Random Forest model on training set
    print("Training...")
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(x_train, y_train)
    
    # Predicting test set results
    print("Testing...")
    y_pred = classifier.predict(x_test)
    
    return (y_pred, y_test)

def learning_analysis(y_pred, y_test):
    y_pred = y_pred.round()
    # Analysis
    diff_vec = []
    diff_over = []
    diff_under = []
    for i in range(0,len(y_pred)):
        ratio_diff = y_pred[i] / y_test[i]
        diff_vec.append(ratio_diff)
        if (ratio_diff < 1):
            diff_under.append(y_pred[i] - y_test[i])
        else:
            diff_over.append(y_pred[i] - y_test[i])
    print("\nPrediction results: ")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1), "\n")
    print("Mean Absolute Error: ")
    print(mean_absolute_percentage_error(y_test, y_pred), "\n")
    print("Absolute Average Difference: ")
    print(statistics.mean(abs(y_pred - y_test)), "\n")
    print("Over-predictions: ")
    print("Average =", statistics.mean(diff_over), ", Count:", len(diff_over), "\n")
    print("Under-predictions: ")
    print("Average =", statistics.mean(diff_under), ", Count:", len(diff_under), "\n")
    x_axis = []
    for i in range(1, len(y_pred) + 1):
        x_axis.append(i)
        
    # Data Visualization
    '''plt.plot(x_axis, y_pred - y_test, label = "Difference")
    plt.title('Prediction vs. Actual', fontsize = 14)
    plt.xlabel('Index', fontsize = 14)
    plt.ylabel('Difference in Prediction vs Actual (in Million $)', fontsize = 14)
    plt.grid(True)
    plt.legend()
    plt.show()'''
    
    plt.figure(figsize=(10,10))
    plt.scatter(y_test, y_pred, c = 'crimson')
    plt.yscale('linear')
    plt.xscale('linear')
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.title('Predictions vs Actual (in Million $)')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
    
    return

def main():
    house_df = read_and_clean('Data/housing_data.csv')
    house_df.to_csv('Data/clean_house.csv', sep=',')
    x_train, x_test, y_train, y_test = train_test_splitting(house_df)
    #y_pred, y_test = decision_tree_learner(x_train, x_test, y_train, y_test)
    #y_pred, y_test = linear_reg_learner(x_train, x_test, y_train, y_test)
    y_pred, y_test = random_forest_learner(x_train, x_test, y_train, y_test)
    learning_analysis(y_pred, y_test)
    
if __name__ == "__main__":
    main()