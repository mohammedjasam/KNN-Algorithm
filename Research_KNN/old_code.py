import pandas
import math

with open("data.csv", 'r') as csvfile:
    data = pandas.read_csv(csvfile)

# The names of all the columns in the data.
# print(data.columns.values)

# Select Lebron James from our dataset
category = data[data["SITE"] == "technology"].iloc[0]


# Choose only the numeric columns (we'll use these to compute euclidean distance)
distance_columns = ['kids', 'say', 'things', 'president', 'diet', 'fitnessliving'
, 'wellparenting', 'tv', 'search', 'crime', 'east', 'digital', 'shows', 'kelly'
, 'wallace', 'november', 'chat', 'facebook', 'messenger', 'find', 'world', 'many'
, 'want', 'videos', 'must', 'watch', 'run', 'according', 'large', 'family', 'life'
, 'read', 'parents', 'twitter', 'school', 'interest', 'much', 'also', 'absolutely'
, 'ever', 'office', 'land', 'thing', 'go', 'could', 'told', 'america', 'march'
, 'presidential', 'campaign', 'end', 'million', 'order', 'get', 'money', 'first'
, 'take', 'time', 'might', 'american', 'times', 'way', 'election', 'children', 'inc'
, 'country', 'leader', 'free', 'high', 'thought', 'know', 'good', 'candidates'
, 'definitely', 'part', 'white', 'house', 'four', 'years', 'vice', 'top', 'young'
, 'really', 'call', 'public', 'service', 'show', 'beyond', 'vote', 'artist', 'model'
, 'someone', 'cancer', 'helping', 'animals', 'asked', 'make', 'better', 'place'
, 'latest', 'share', 'comments', 'health', 'hillary', 'clinton', 'female', 'even'
, 'actually', 'chance', 'lady', 'content', 'pay', 'card', 'save', 'enough'
, 'reverse', 'risk', 'paid', 'partner', 'cards', 'around', 'next', 'generation'
, 'big', 'network', 'system', 'rights', 'reserved', 'terms', 'mexican', 'meeting'
, 'trump', 'january', 'mexico', 'different', 'route', 'border', 'immigrants'
, 'trying', 'donald', 'wall', 'billion', 'signs', 'executive', 'actions'
, 'building', 'along', 'southern', 'nowstory', 'believe', 'fruitless', 'thursday'
, 'set', 'week', 'plan', 'tuesday', 'something', 'recently', 'wednesday', 'needed'
, 'tweet', 'trade', 'nafta', 'massive', '@realdonaldtrump', 'jobs', 'companies'
, 'remarks', 'gathering', 'congressional', 'republicans', 'planned', 'together'
, 'unless', 'senate', 'gop', 'lawmakers', 'security', 'national', 'problem'
, 'illegal', 'immigration', 'see', 'need', 'statement', 'back', 'two', 'leaders'
, 'last', 'year', 'days', 'called', 'action', 'begin', 'process', 'announced'
, 'move', 'level', 'foreign', 'representatives', 'come', 'since', 'officials'
, 'including', 'staff', 'minister', 'government', 'team', 'car', 'department'
, 'homeland', 'work', 'help', 'united', 'states', 'forces', 'number', 'officers'
, 'visit', 'try', 'able', 'related', 'monday', 'migrants', 'home', 'city'
, 'conversation', 'made']

def euclidean_distance(row):
    """
    A simple euclidean distance function
    """
    inner_value = 0
    for k in distance_columns:
        inner_value += (row[k] - category[k]) ** 2
    return math.sqrt(inner_value)

# Find the distance from each player in the dataset to lebron.
technology_distance = data.apply(euclidean_distance, axis=1)

# print(technology_distance)

# Select only the numeric columns from the NBA dataset
data_numeric = data[distance_columns]

# Normalize all of the numeric columns
data_normalized = (data_numeric - data_numeric.mean()) / data_numeric.std()


# print(data_normalized)

from scipy.spatial import distance

# Fill in NA values in nba_normalized
data_normalized.fillna(0, inplace=True)


# Find the normalized vector for lebron james.
technology_normalized = data_normalized[data["SITE"] == "technology1"]

# print(technology_normalized)

# Find the distance between lebron james and everyone else.
euclidean_distances = data_normalized.apply(lambda row: distance.euclidean(row, technology_normalized), axis=1)

# Create a new dataframe with distances.
distance_frame = pandas.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
distance_frame.sort("dist", inplace=True)
# Find the most similar player to lebron (the lowest distance to lebron is lebron, the second smallest is the most similar non-lebron player)
second_smallest = distance_frame.iloc[1]["idx"]
most_similar_to_technology = data.loc[int(second_smallest)]["SITE"]


print(most_similar_to_technology)


import random
from numpy.random import permutation

# Randomly shuffle the index of nba.
random_indices = permutation(data.index)
# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
test_cutoff = math.floor(len(data)/3)
# Generate the test set by taking the first 1/3 of the randomly shuffled indices.
test = data.loc[random_indices[1:test_cutoff]]
# Generate the train set with the rest of the data.
train = data.loc[random_indices[test_cutoff:]]


# The columns that we will be making predictions with.
x_columns = ['kids', 'say', 'things', 'president', 'diet', 'fitnessliving'
, 'wellparenting', 'tv', 'search', 'crime', 'east', 'digital', 'shows', 'kelly'
, 'wallace', 'november', 'chat', 'facebook', 'messenger', 'find', 'world', 'many'
, 'want', 'videos', 'must', 'watch', 'run', 'according', 'large', 'family', 'life'
, 'read', 'parents', 'twitter', 'school', 'interest', 'much', 'also', 'absolutely'
, 'ever', 'office', 'land', 'thing', 'go', 'could', 'told', 'america', 'march'
, 'presidential', 'campaign', 'end', 'million', 'order', 'get', 'money', 'first'
, 'take', 'time', 'might', 'american', 'times', 'way', 'election', 'children', 'inc'
, 'country', 'leader', 'free', 'high', 'thought', 'know', 'good', 'candidates'
, 'definitely', 'part', 'white', 'house', 'four', 'years', 'vice', 'top', 'young'
, 'really', 'call', 'public', 'service', 'show', 'beyond', 'vote', 'artist', 'model'
, 'someone', 'cancer', 'helping', 'animals', 'asked', 'make', 'better', 'place'
, 'latest', 'share', 'comments', 'health', 'hillary', 'clinton', 'female', 'even'
, 'actually', 'chance', 'lady', 'content', 'pay', 'card', 'save', 'enough'
, 'reverse', 'risk', 'paid', 'partner', 'cards', 'around', 'next', 'generation'
, 'big', 'network', 'system', 'rights', 'reserved', 'terms', 'mexican', 'meeting'
, 'trump', 'january', 'mexico', 'different', 'route', 'border', 'immigrants'
, 'trying', 'donald', 'wall', 'billion', 'signs', 'executive', 'actions'
, 'building', 'along', 'southern', 'nowstory', 'believe', 'fruitless', 'thursday'
, 'set', 'week', 'plan', 'tuesday', 'something', 'recently', 'wednesday', 'needed'
, 'tweet', 'trade', 'nafta', 'massive', '@realdonaldtrump', 'jobs', 'companies'
, 'remarks', 'gathering', 'congressional', 'republicans', 'planned', 'together'
, 'unless', 'senate', 'gop', 'lawmakers', 'security', 'national', 'problem'
, 'illegal', 'immigration', 'see', 'need', 'statement', 'back', 'two', 'leaders'
, 'last', 'year', 'days', 'called', 'action', 'begin', 'process', 'announced'
, 'move', 'level', 'foreign', 'representatives', 'come', 'since', 'officials'
, 'including', 'staff', 'minister', 'government', 'team', 'car', 'department'
, 'homeland', 'work', 'help', 'united', 'states', 'forces', 'number', 'officers'
, 'visit', 'try', 'able', 'related', 'monday', 'migrants', 'home', 'city'
, 'conversation', 'made']
# The column that we want to predict.
y_column = ["facebook"]

from sklearn.neighbors import KNeighborsRegressor
# Create the knn model.
# Look at the five closest neighbors.
knn = KNeighborsRegressor(n_neighbors=5)
# print(knn)

# Fit the model on the training data.
knn.fit(train[x_columns], train[y_column])
# Make point predictions on the test set using the fit model.
predictions = knn.predict(test[x_columns])

# print(predictions)

# Get the actual values for the test set.
actual = test[y_column]

# Compute the mean squared error of our predictions.
mse = (((predictions - actual) ** 2).sum()) / len(predictions)

print(mse)
