import nltk
nltk.download("popular")
import json
import warnings
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Ignore warnings
warnings.filterwarnings('ignore')

# Loading the data
print('Reading the data...')
x_train = []
x_test = []
y_train = []
y_test = []
with open('../Data/Interim/x_train.json', 'r', encoding='utf-8') as x_train_file:
  x_train = json.load(x_train_file)
with open('../Data/Interim/x_test.json', 'r', encoding='utf-8') as x_test_file:
  x_test = json.load(x_test_file)
with open('../Data/Interim/y_train.json', 'r', encoding='utf-8') as y_train_file:
  y_train = json.load(y_train_file)
with open('../Data/Interim/y_test.json', 'r', encoding='utf-8') as y_test_file:
  y_test = json.load(y_test_file)

# Vectorization
print('Applying TF-IDF vectorization')
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
vectorizer.fit(x_train + x_test)
x_train_tfidf = vectorizer.transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)
k = int(0.3*x_train_tfidf.shape[1])

# Selection
print('Selecting the best features')
selector = SelectKBest(score_func=chi2, k=k)
x_train_tfidf = selector.fit_transform(x_train_tfidf, y_train)
x_test_tfidf = selector.transform(x_test_tfidf)

# Pos tagging
x_train_tfidf = pd.DataFrame(data=x_train_tfidf.toarray())
x_test_tfidf = pd.DataFrame(data=x_test_tfidf.toarray())

pos_array = []

def n_adjs(pos_tags):
  number = 0
  for tag in pos_tags:
    if tag[1] == 'JJ': # Adjectives
      number += 1
  return number

print('\tPos tagging the data for x_train...')
for voc in tqdm(x_train):
  pos_tags = nltk.pos_tag(voc.split(' '))
  pos_array.append(n_adjs(pos_tags))

print('\tPos tagging the data for x_test...')
pos_test = []
for voc in tqdm(x_test):
  pos_tags = nltk.pos_tag(voc.split(' '))
  pos_test.append(n_adjs(pos_tags))

x_train_tfidf['n_adjs'] = pos_array
x_test_tfidf['n_adjs'] = pos_test

# Undersampling
print('Undersampling the data')
x_train_tfidf['label'] = y_train
n = int(22111 * 0.3) # undersampling rate
msk = x_train_tfidf.groupby('label')['label'].transform('size') >= n
x_train_tfidf = pd.concat((x_train_tfidf[msk].groupby('label').sample(n=n), x_train_tfidf[~msk]), ignore_index=True)
print('\tFinal undersampled data:')
print(x_train_tfidf['label'].value_counts())

# Naive Bayes
print('Training the Naive Bayes model')
mngb_model = MultinomialNB().fit(x_train_tfidf.drop('label', axis=1), x_train_tfidf['label'])
preds = mngb_model.predict(x_test_tfidf)
print(f'\t --> F1 Score for the model: {f1_score(y_test, preds, average="micro")}')
print(f'\t --> Precision for the model: {precision_score(y_test, preds, average="micro")}')
print(f'\t --> Recall for the model: {recall_score(y_test, preds, average="micro")}')

# Decision Tree
print('Training the Decision Tree model')
dt_model = DecisionTreeClassifier().fit(x_train_tfidf.drop('label', axis=1), x_train_tfidf['label'])
preds = dt_model.predict(x_test_tfidf)
print(f'\t --> F1 Score for the model: {f1_score(y_test, preds, average="micro")}')
print(f'\t --> Precision for the model: {precision_score(y_test, preds, average="micro")}')
print(f'\t --> Recall for the model: {recall_score(y_test, preds, average="micro")}')

# Voting Classifier
print('Training the Voting Classifier model')
## kNN
n_neighbors = 16
weights = 'uniform'
knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric='minkowski', p=1)
## Voting
voting_model = VotingClassifier(estimators=[('mnb', mngb_model), ('dt', dt_model), ('knn', knn_model)], voting='soft')
voting_model.fit(x_train_tfidf.drop('label', axis=1), x_train_tfidf['label'])
preds = voting_model.predict(x_test_tfidf)
print(f'\t --> F1 Score for the model: {f1_score(y_test, preds, average="micro")}')
print(f'\t --> Precision for the model: {precision_score(y_test, preds, average="micro")}')
print(f'\t --> Recall for the model: {recall_score(y_test, preds, average="micro")}')