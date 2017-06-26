import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cross_validation import train_test_split
#to slip data into training and testing

from sklearn.naive_bayes import MultinomialNB


count_vector = CountVectorizer() # LowerCase; elliminate puncuations; elliminate stop_words

df = pd.read_table('hi', sep='\t', header=None, names=['label', 'sms_msg'])

df['label'] = df.label.map({'ham':0, 'spam': 1})
print(df.shape) #returns size of rows and columns
#print(df.head(n=5)) #Returns rows and columns
#print(count_vector)

#Fit your document dataset to the CountVectorizer object you have created using fit(), and get the list of words which have been categorized as features using the get_feature_names() method.

X_train, X_test, Y_train, Y_test = train_test_split(df['sms_msg'], df['label'], random_state=1)
print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# Divide into testing and training

training_data = count_vector.fit_transform(X_train)
# Fit the training data and return the matrix with no puncuations, All to lower and no stop words, with frequency

testing_data = count_vector.transform(X_test)
# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, Y_train)
# Training

predictions = naive_bayes.predict(testing_data)
# Testing


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(Y_test, predictions)))
print('Precision score: ', format(precision_score(Y_test, predictions)))
print('Recall score: ', format(recall_score(Y_test, predictions)))
print('F1 score: ', format(f1_score(Y_test, predictions)))
