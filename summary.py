# https://docs.google.com/document/d/12FRFEYpchiGuSa2VQbrbnSMRAATVPT0bgSE_YblraxY/edit

# Libraries and packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# ------------------------------------------------
####  EDA  ####

# Read csv
df = pd.read_csv("df.txt")

# Explore df
df.dtypes
df.info()
df["Churn?"].value_counts()
df.isnull()
df.shape
df.describe()


# ------------------------------------------------
####  KNN Estimation  ####

# KNN without normalization and gridsearch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

array = df.values
X = array[:,:-1]
y = array[:,-1]

test_size = 0.20
seed = 42
num_folds = 10
scoring = 'neg_mean_squared_error'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
msg = "KNN: %f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

# KNN with normalization and gridsearch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

array = df.values
X = array[:,:-1]
y = array[:,-1]

test_size = 0.20
seed = 42
num_folds = 10
scoring = 'neg_mean_squared_error'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

scaler = StandardScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)

k_values = np.arange(1,22)
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

scaler = StandardScaler().fit(X_train)
rescaledX_train = scaler.transform(X_train)
model = KNeighborsRegressor(n_neighbors=2)
model.fit(rescaledX_train, y_train)

rescaledX_test = scaler.transform(X_test)
estimates = model.predict(rescaledX_test)
mean_squared_error(y_test, estimates)

# KNN classifier with confusion matrix
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform')
clf.fit(X, y)
classes = clf.predict(X)

confusion_matrix(y, classes)

# ------------------------------------------------
####  Decision Tree  ####

# !pip install graphviz
# !pip install dmba
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from dmba import classificationSummary

xlsx = pd.ExcelFile('UniversalBank.xlsx')
bank_df = pd.read_excel(xlsx, 'Data')

bank_df.columns = bank_df.columns.str.replace(' ','')
bank_df.drop(columns=['ID','ZIPCode'], inplace=True)

X = bank_df.drop(columns=['PersonalLoan'])
y = bank_df['PersonalLoan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

fullClassTree = DecisionTreeClassifier(random_state=1)
fullClassTree.fit(X_train, y_train)
y_predicted = fullClassTree.predict(X_test)
accuracy_score(y_test, y_predicted)
classificationSummary(y_train, fullClassTree.predict(X_train))
classificationSummary(y_test, fullClassTree.predict(X_test))

fullClassTree.tree_.max_depth
fullClassTree.tree_.node_count
fullClassTree.get_params()

export_graphviz(fullClassTree, out_file='fullClassTree.dot', 
                feature_names=X_train.columns)

# Decision tree Grid search
param_grid = {
    'max_depth': [10, 20, 30, 40],
    'min_samples_split': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

gridSearch = GridSearchCV(DecisionTreeClassifier(random_state=1),
                         param_grid, cv=5)

gridSearch.fit(X_train, y_train)
print('Initial score: ', gridSearch.best_score_)
print('Initial parameters: ', gridSearch.best_params_)
bestClassTree = gridSearch.best_estimator_
classificationSummary(y_train, bestClassTree.predict(X_train))

# ------------------------------------------------
####  ANN Neural Network  ####

# !pip install Keras
# !pip install tensorflow

from keras.models import Sequential
from sklearn import preprocessing
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

np.random.seed(seed)

dataset = pd.read_csv("pima-indians-diabetes.csv", header=None)

scaler = preprocessing.MinMaxScaler()
scaled_dataset= scaler.fit_transform(dataset)

X = scaled_dataset[:,:8]
y = scaled_dataset[:,8]

model = Sequential()

model.add(Dense(12, input_dim=8, kernel_initializer='uniform', 
          activation='sigmoid'))

model.add(Dense(1, kernel_initializer='uniform', 
          activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
              metrics=['accuracy'])

model.fit(X, y, epochs=50, batch_size=10, verbose=2)

# Grid search ANN
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def create_model(neurons=1):
    model = Sequential()
    model.add(Dense(neurons, input_dim=8, kernel_initializer='uniform', 
                    activation='sigmoid'))
    model.add(Dense(1, kernel_initializer='uniform', 
                    activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy']) 
    return model

model = KerasClassifier(build_fn=create_model, epochs=100, 
                        batch_size=10, verbose=2)

neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
grid_result.cv_results_

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with %r" % (mean, stdev, param))

# ------------------------------------------------
####  Text Mining  ####

# !pip install wordcloud

from string import punctuation
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud
from keras.models import Sequential
from keras.layers import Dense

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = text.split()
    table = str.maketrans('','', punctuation)
    tokens = [word.translate(table) for word in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if not word in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)

sentences = pd.DataFrame(["First we consider the initial spreadsheet model.",
                         "Then we consider another model, not the initial one.",
                         "First we consider the initial spreadsheet model, and then we consider another model."],
                        columns = ["sentence"])
pd.set_option('max_colwidth', 400)

sentences.sentence = sentences.sentence.apply(clean_text)
sentence_list = list(sentences.sentence)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence_list)
tokenizer.word_index

col_names = list(tokenizer.word_index)
word_bag = tokenizer.texts_to_matrix(sentence_list, mode='binary')
wordbag_df = pd.DataFrame(word_bag)
wordbag_df.columns = ["dummy"] + col_names
wordbag_df

modes = ['binary' ,'count', 'tfidf', 'freq']
for mode in modes:
    word_bag = tokenizer.texts_to_matrix(sentence_list, mode=mode)
    print(mode)
    print(word_bag)

# Data mining on a df
movie_df = pd.read_csv("MovieReviews.csv")
movie_df.Review = movie_df.Review.apply(clean_text)

review_list = list(movie_df.Review)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(review_list)
tokenizer.word_index
col_names = list(tokenizer.word_index)

word_bag = tokenizer.texts_to_matrix(review_list, mode='binary')
wordbag_df = pd.DataFrame(word_bag)
wordbag_df.columns = ["dummy"] + col_names

wordbag_df.sum()
keywords = pd.DataFrame(wordbag_df.sum().sort_values(ascending=False)).head(50)

cloud_df = keywords.reset_index()
cloud_df.columns = ["word", "count"]
tuples = [tuple(x) for x in cloud_df.values]
cloud_of_words = WordCloud(background_color='white').
generate_from_frequencies(dict(tuples))

plt.imshow(cloud_of_words, interpolation='bilinear')
plt.axis('off')


tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)

Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
n_words = Xtrain.shape[1]
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

positive_lines = review_list[:900]
negative_lines = review_list[1000:1900]
docs = negative_lines + positive_lines

Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

model = Sequential()
model.add(Dense(50, input_dim=n_words, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(Xtrain, ytrain, epochs=50, verbose=2)
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))

# ------------------------------------------------
####  K-Means Clustering  ####

from sklearn.cluster import KMeans
from numpy import array
from numpy import hstack
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)

pipeline = make_pipeline(kmeans)
pipeline.fit(X)
labels = pipeline.predict(X)
df = pd.DataFrame({'labels': labels, 'varieties': y})
ct = pd.crosstab(df['labels'], df['varieties'])

# training split and gridsearch
test_size = 0.3
num_folds = 10

df_K_p2=df

df_K_p2.reset_index(drop=True, inplace=True)
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(df_K_p2[['Cluster']]).toarray())
enc_df.columns = enc.get_feature_names_out(['Cluster'])

joined_df = df_K_p2.join(enc_df)
joined_df.drop(columns=['Cluster'], inplace=True)

columns = ['Population', 'Avg_HH_size', 'Total_before_2005',
        'HH_apt_LH_rise', 'HH_dwelling_other',
       'HH_not_tenure', 'Dwellings_renter', 'Cluster_0',
       'Cluster_1', 'Median_income']
joined_df = joined_df[columns]

joined_array = joined_df.to_numpy()
X = joined_array[:,:-1]
y = joined_array[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)

scaler = StandardScaler().fit(X_train[:,:-3])
rescaled_X_train = scaler.transform(X_train[:,:-3])
rescaled_X_train = np.concatenate((rescaled_X_train, X_train[:,-3:]), axis=1)

knn_scoring = 'neg_mean_absolute_error'
k_values = np.arange(1,50)
knn_param_grid = dict(n_neighbors = k_values, weights = ["distance", "uniform"])

knn_model = KNeighborsRegressor()
knn_grid = GridSearchCV(estimator = knn_model, param_grid = knn_param_grid, scoring = knn_scoring, cv= num_folds)
knn_grid_result = knn_grid.fit(rescaled_X_train, y_train)

print(f"Best: {knn_grid_result.best_score_} {knn_grid_result.best_params_}" )

grid_result = knn_grid_result

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

best_n_neighbour = knn_grid_result.best_params_['n_neighbors']
best_weights = knn_grid_result.best_params_['weights']


knn_trained_model = KNeighborsRegressor(n_neighbors = best_n_neighbour, weights=best_weights)
knn_trained_model.fit(rescaled_X_train,y_train)

rescaled_X_test = scaler.transform(X_test[:,:-3])
rescaled_X_test = np.concatenate((rescaled_X_test, X_test[:,-3:]), axis=1)

y_pred = knn_trained_model.predict(rescaled_X_test)
mean_absolute_error(y_test, y_pred)