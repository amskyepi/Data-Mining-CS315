import numpy as np

# Programming Assignment 3
# CS 315
# Written by Amethyst Skye

def sign(feature, weight):
    if np.dot(feature, weight) >= 0:
        return 1
    else:
        return 0

# Purpose: Binary classifier which will take 5 files containing:
# 1) stop list
# 2) training data
# 3) testing data
# 4) training labels
# 5) testing labels
# The classifier will compute the number of mistakes during each iteration, the training
# and testing accuracy after each iteration, the training/testing accuracy after 20
# iterations with a standard perceptron and average perceptron.
def binary_classifier(f_stop_list, f_train_data, f_test_data, f_train_labels, f_test_labels):
    # A list of words to ignore
    with open(f_stop_list, 'r') as stoplist_file:
        stopwords = stoplist_file.read().split()
    stoplist_file.close()
    
    # Read training data and filter out words that are in the stopwords list
    with open(f_train_data, 'r') as traindata_file:
        words = traindata_file.read().split()
        words = np.unique(np.array(words))
        word_list = list(filter(lambda x: x not in stopwords, words))
        traindata_file.seek(0)
        features = []
        
        # For each line, classify each feature if a word is found in word_list
        for line in traindata_file:
            message = line.split()
            feature = []
            for word in word_list:
                if word in message:
                    feature.append(1)
                else:
                    feature.append(0)
            features.append(feature)
    traindata_file.close()
    
    # Classify testing data 
    with open(f_test_data, 'r') as testdata_file:
        test_features = []
        for line in testdata_file:
            message = line.split()
            test_feature = []
            for word in word_list:
                if word in message:
                    test_feature.append(1)
                else:
                    test_feature.append(0)
            test_features.append(test_feature)
    testdata_file.close()
    train_label = []
    
    # Training labels
    with open(f_train_labels, 'r') as trainlabel_file:
        for line in trainlabel_file:
            train_label.append(line)
    trainlabel_file.close()
    test_label = []
    
    # Testing labels
    with open(f_test_labels, 'r') as testlabel_file:
        for line in testlabel_file:
            test_label.append(line)
    testlabel_file.close()
    
    # Assign weights
    weight = []
    for word in word_list:
        weight.append(0)
    avg_weight = weight
    
    # Iterate through features and count mistakes
    train_accuracy = {}
    mistakes = {}
    test_accuracy = {}
    iter_count = 1
    c = 1
    while(iter_count <= 20):
        i = 0
        train_mistakes = 0
        for feature in features:
            if sign(feature, weight) != int(train_label[i]):
                train_mistakes += 1
                if sign(feature, weight) < int(train_label[i]):
                    feature_new = np.dot(feature, 1)
                    feature_avg = np.dot(feature_new, c)
                    avg_weight = np.add(avg_weight, feature_avg)
                    weight = np.add(weight, feature_new)
                else:
                    feature_new = np.dot(feature, -1)
                    feature_avg = np.dot(feature_new, c)
                    avg_weight = np.add(avg_weight, feature_avg)
                    weight = np.add(weight, feature_new)
            c += 1
            i += 1
        i = 0
        mistakes[iter_count] = train_mistakes
        train_accuracy[iter_count] = (1 - (train_mistakes / 322)) * 100
        test_mistakes = 0
        
        # Iterate test features to find mistakes
        for test_feature in test_features:
            if sign(test_feature, weight) != int(test_label[i]):
                test_mistakes += 1
            i += 1
        test_accuracy[iter_count] = (1-(test_mistakes/len(test_features)))*100
        iter_count += 1
    avg = weight - (1 / c) * avg_weight
    
    # Calculate training accuracy for standard perceptron
    i = 0
    train_correct = 0
    train_avg = 0
    for feature in features:
        if sign(feature, weight) == int(train_label[i]):
            train_correct += 1
        if sign(feature, avg) == int(train_label[i]):
            train_avg += 1
        i += 1
        
    # Calculate testing accuracy for averaged perceptron
    i = 0
    test_correct = 0
    test_avg = 0
    for test_feature in test_features:
        if sign(test_feature, weight) == int(test_label[i]):
            test_correct += 1
        if sign(test_feature, avg) == int(test_label[i]):
            test_avg += 1
        i += 1
        
    # Write output to a file
    with open("output.txt", 'w') as out_file:
        out_file.write("Fortune Cookie Classifier\n\n")
        
        # Mistakes
        for key in mistakes.keys():
            string = "iteration-" + str(key) + " " + str(mistakes[key]) + "\n"
            out_file.write(string)
        out_file.write("\n")
        
        # Training/Testing accuracy
        for key in train_accuracy.keys():
            string = "iteration-" + str(key) + " " + str(round(train_accuracy[key], 2)) + "% " + str(round(test_accuracy[key], 2)) + "%\n"
            out_file.write(string)
        out_file.write("\n")
        
        # Training accuracy for standard perceptron
        string = str(round(train_correct / 322 * 100, 2)) + "% " + str(round(train_avg / 322 * 100, 2)) + "%\n\n"
        out_file.write(string)
        
        # Testing accuracy for averaged perceptron
        string = str(round(test_correct / 101 * 100, 2)) + "% " + str(round(test_avg / 101 * 100, 2)) + "%\n\n"
        out_file.write(string)
    out_file.close()
    return

# Purpose: OCR classifier which will take 2 files containing:
# 1) training data
# 2) testing data
# The classifier will predict the corrosponding letter given a handwritten character.
# We will compute the number of mistakes during each iteration, the training
# and testing accuracy after each iteration, the training/testing accuracy after 20
# iterations with a standard perceptron and average perceptron.
def ocr_classifier(f_training, f_testing):
    # Read training data
    with open(f_training, 'r') as traindata:
        features = {}
        i = 1
        vowels = ["a", "e", "i", "o", "u"]
        # Clean data
        for line in traindata:
            if line.strip():
                letter = line.split()
                feature = letter[1]
                feature = feature.replace("im", "")
                feature_list = []
                for char in feature:
                    feature_list.append(float(char))
                letter = letter[2]
                if letter in vowels:
                    letter = 0
                else:
                    letter = 1
                features[i] = (np.array(feature_list),letter)
                i += 1
    traindata.close()
    # Read testing data
    with open(f_testing, 'r') as testdata:
        test_features = {}
        i = 1
        vowels = ["a","e","i","o","u"]
        # Clean test data
        for line in testdata:
            if line.strip():
                letter = line.split()
                feature = letter[1].replace("im", "")
                feature_list = []
                for char in feature:
                    feature_list.append(float(char))
                letter = letter[2]
                if letter in vowels:
                    letter = 0
                else:
                    letter = 1
                test_features[i] = (np.array(feature_list),letter)
                i += 1
    testdata.close()
    
    # Assign weights
    weight = []
    weight_num = 0
    while weight_num < features[1][0].size:
        weight.append(0)
        weight_num += 1
    avg_weight = weight
    
    # Iterate through training features and find mistakes
    accuracy = {}
    mistakes = {}
    test_accuracy = {}
    iter_count = 1
    c = 1
    while(iter_count <= 20):
        train_mistakes=0
        for feature in features.items():
            if sign(feature[1][0],weight) != int(feature[1][1]):
                train_mistakes+=1
                if sign(feature[1][0],weight) < int(feature[1][1]):
                    feature_new = np.dot(feature[1][0], 1)
                    feature_avg = np.dot(feature_new, c)
                    avg_weight = np.add(avg_weight, feature_avg)
                    weight = np.add(weight, feature_new)
                else:
                    feature_new = np.dot(feature[1][0], -1)
                    feature_avg = np.dot(feature_new, c)
                    avg_weight = np.add(avg_weight, feature_avg)
                    weight = np.add(weight, feature_new)
            c += 1
        mistakes[iter_count] = train_mistakes
        accuracy[iter_count] = (1 - (train_mistakes / len(features.keys()))) * 100
        
        # Iterate through testing features and find mistakes
        test_mistakes = 0
        for test_feature in test_features.items():
            if sign(test_feature[1][0],weight) != int(test_feature[1][1]):
                test_mistakes += 1
        test_accuracy[iter_count] = (1 - (test_mistakes / len(test_features.keys()))) * 100
        iter_count += 1
    avg = weight - np.dot((1 / c), avg_weight)
    
    # Training accuracy for standard preceptron
    correct = 0
    correct_avg = 0
    for feature in features.items():
        if sign(feature[1][0], weight) == int(feature[1][1]):
            correct+=1
        if sign(feature[1][0], avg) == int(feature[1][1]):
            correct_avg+=1
    
    # Testing accuracy for averaged preceptron   
    test_correct = 0
    test_avg = 0
    for test_feature in test_features.items():
        if sign(test_feature[1][0], weight) == int(test_feature[1][1]):
            test_correct += 1
        if sign(test_feature[1][0], avg) == int(test_feature[1][1]):
            test_avg += 1
            
    # Write output file
    with open("output.txt", 'a') as out:
        out.write("OCR Classifier\n\n")
        
        # Number of mistakes per iteration
        for key in mistakes.keys():
            string = "iteration-" + str(key) + " " + str(mistakes[key]) + "\n"
            out.write(string)
        out.write("\n")
        
        # Accuracy per iteration
        for key in accuracy.keys():
            string = "iteration-" + str(key) + " " + str(round(accuracy[key],2)) + "% " + str(round(test_accuracy[key], 2)) + "%\n"
            out.write(string)
        out.write("\n")
        
        # Training accuracy for standard preceptron 
        string = str(round(correct/len(features.keys())*100,2)) + "% " + str(round(correct_avg/len(features.keys())*100,2)) + "%\n\n"
        out.write(string)
        
        # Testing accuracy for averaged preceptron 
        string = str(round(test_correct/len(test_features.keys())*100,2)) + "% " + str(round(test_avg/len(test_features.keys())*100,2)) + "%"
        out.write(string)
    out.close()
    return 
    
#---------------------------------------------------------------------------------------#

binary_classifier("fortune-cookie-data/stoplist.txt",
                    "fortune-cookie-data/traindata.txt",
                    "fortune-cookie-data/testdata.txt",
                    "fortune-cookie-data/trainlabels.txt",
                    "fortune-cookie-data/testlabels.txt")

ocr_classifier("OCR-data/ocr_train.txt", 
                        "OCR-data/ocr_test.txt")
