import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
tennis_stats = pd.read_csv('tennis_stats.csv')
df = pd.DataFrame(tennis_stats)

id_data = df[['Player', 'Year']]

features = df[['BreakPointsOpportunities']]

outcomes = df[['Wins','Losses','Winnings','Ranking']]


# perform exploratory analysis here:
plt.scatter(features['BreakPointsOpportunities'], outcomes['Wins'])
plt.xlabel('Break Point Opportunities')
plt.ylabel('Wins')
plt.show()

plt.scatter(df['Aces'], df['Winnings'])
plt.xlabel('Aces')
plt.ylabel('Winnings')
plt.show()

## perform single feature linear regressions here:

features = df[['FirstServeReturnPointsWon']]
outcomes = df[['Winnings']]

#split the data between training and test set
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features, outcomes, train_size = 0.8, random_state = 6)

#creating and training the testing model
model = LinearRegression()
model.fit(features_train, outcomes_train)
print('Model score using first serve opportunities: ')
print(model.score(features_test, outcomes_test))

#using model to predict
prediction = model.predict(features_test)

#plotting data
plt.scatter(outcomes_test, prediction, alpha=0.4)
plt.xlabel('First Serve Return Points Won')
plt.ylabel('Winnings')
plt.show()

#using 'BreakPointsOpportunities' as the feature to predict 'Winnings'
features = df[['BreakPointsOpportunities']]
outcomes = df[['Winnings']]

#split the data between training and test set
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features, outcomes, train_size = 0.8, random_state = 6)

#creating and training the testing model
model = LinearRegression()
model.fit(features_train, outcomes_train)
print('Model score using break point opportunities: ')
print(model.score(features_test, outcomes_test))

#using model to predict
prediction = model.predict(features_test)

plt.scatter(outcomes_test, prediction)
plt.xlabel('Break Point Opportunities')
plt.ylabel('Winnings')
plt.show()

## perform two feature linear regressions here:
#using two features
features = df[['BreakPointsOpportunities', 'FirstServeReturnPointsWon']]
outcomes = df[['Winnings']]

#split the data between training and test set
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features, outcomes, train_size = 0.8, test_size = 0.2, random_state = 6)

#creating and training the testing model
model = LinearRegression()
model.fit(features_train, outcomes_train)
print('Model score using BOTH break point opportunities and first serve return point wins: ')
print(model.score(features_test, outcomes_test))

#using model to predict
prediction = model.predict(features_test)

plt.scatter(features_test['BreakPointsOpportunities'], prediction)
plt.xlabel('Break Point Opportunities')
plt.ylabel('Winnings')
plt.show()


## perform multiple feature linear regressions here:


# Try different combinations, e.g.:
features = df[['Aces', 'FirstServePointsWon', 'BreakPointsConverted']]

# ...repeat model training and evaluation
#split the data between training and test set
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features, outcomes, train_size = 0.8, test_size = 0.2, random_state = 6)

#creating and training the testing model
model = LinearRegression()
model.fit(features_train, outcomes_train)
print('Model score using all features: ')
print(model.score(features_test, outcomes_test))

#using model to predict
prediction = model.predict(features_test)

#plotting data
plt.scatter(features_test['Aces'], prediction)
plt.xlabel('Aces')
plt.ylabel('Winnings')
plt.title('Aces vs Winnings')
plt.show()


#using all the features in the data set that can have a impact on a match
features = df[['Aces','DoubleFaults', 'FirstServe','FirstServePointsWon','SecondServePointsWon','BreakPointsFaced', 'BreakPointsSaved','ServiceGamesPlayed','ServiceGamesWon','TotalServicePointsWon','FirstServeReturnPointsWon','SecondServeReturnPointsWon','BreakPointsOpportunities','BreakPointsConverted','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon','TotalPointsWon']]
outcomes = df[['Winnings']]

#split the data between training and test set
features_train, features_test, outcomes_train, outcomes_test = train_test_split(features, outcomes, train_size = 0.8, test_size = 0.2, random_state = 6)

#creating and training the testing model
model = LinearRegression()
model.fit(features_train, outcomes_train)
print('Model score using all features: ')
print(model.score(features_test, outcomes_test))

#using model to predict
prediction = model.predict(features_test)

