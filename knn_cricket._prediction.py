import warnings
warnings.filterwarnings('ignore')


from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')


path_to_matches="/content/drive/MyDrive/Colab Notebooks/Cricket Prediction/matches.csv"

matches=pd.read_csv(path_to_matches)
matches.head(5)

matches.tail(5)

matches.describe()

matches.info()

matches.shape

matches.dtypes

matches=matches.sort_values("season", ascending=True)
matches

matches=matches[['team1','team2','season','toss_winner','toss_decision','winner']]
matches

matches.isnull().sum()

matches.isna().sum()

matches = matches.dropna(axis=0, how='any')
matches

matches.isna().sum()

matches['winner'] = matches['winner'].fillna('DRAW')

matches.info()



matches.team1.value_counts()

matches["team1"]=matches['team1'].replace({'Rising Pune Supergiants':'Rising Pune Supergiant'})

matches["team1"]=matches['team1'].replace({'Pune Warriors':'Rising Pune Supergiant'})

matches.team1.value_counts()

teamMapper = {"Mumbai Indians": 0, "Chennai Super Kings": 1, "Kings XI Punjab":2, "Royal Challengers Bangalore" : 3,
"Kolkata Knight Riders" : 4,"Delhi Daredevils" : 5,"Rajasthan Royals" : 6,"Sunrisers Hyderabad" : 7,"Deccan Chargers" : 8,
"Rising Pune Supergiant" : 9,"Gujarat Lions" : 10, "Kochi Tuskers Kerala":11,'DRAW': 12}

for dataset in [matches]:
    dataset['team1'] = dataset['team1'].map(teamMapper)

matches.team1.value_counts()

for dataset in [matches]:
    dataset['team2'] = dataset['team2'].map(teamMapper)
    dataset['toss_winner'] = dataset['toss_winner'].map(teamMapper)
    dataset['winner'] = dataset['winner'].map(teamMapper)

matches.head(5)

matches=matches.drop(['toss_decision'],axis=1)

matches.corr()["winner"]

sns.lmplot(x='team2',y='winner',data=matches,fit_reg=True, ci=None)

sns.lmplot(x='toss_winner', y='winner', data=matches, fit_reg=True, ci=None)

matches.plot.hist(y="winner")

matches[matches.isnull().any(axis=1)].head(20)

matches = matches.dropna()

matches.shape

matches.head(20)

matches["winner"] = matches["winner"].round()

matches.head(20)

X = matches.drop(['winner'], axis = 1)
y = matches['winner']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train, y_train)

y_prediction_knn = clf_knn.predict(X_test)

accuracy_score_knn = clf_knn.score(X_train, y_train)*100
print (accuracy_score_knn)

accuracy_score_knn = clf_knn.score(X_test, y_test)*100
print (accuracy_score_knn)


print('Recall: ',metrics.recall_score(y_test, y_prediction_knn, zero_division=1,average='weighted'))
print('Precision:',metrics.precision_score(y_test, y_prediction_knn, zero_division=1, average='weighted'))
print('CL Report:',metrics.classification_report(y_test, y_prediction_knn, zero_division=1))

y_pred_proba = clf_knn.predict_proba(X_test)

classes = np.unique(y_train)

# Ensure y_test is of the same type as y_train before binarizing
y_test = y_test.astype(y_train.dtype)
y_test_binarized = label_binarize(y_test, classes=classes)

fpr = dict()
tpr = dict()
roc_auc = dict()
reverse_teamMapper = {v: k for k, v in teamMapper.items()}

# Calculate ROC curve and AUC for each class
for i in range(len(classes)):
    # Use y_test_binarized[:, i] to get the true binary labels for class i
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10,5))
lw = 2
for i in range(len(classes)):
    # Get the team name for the current class index 'i'
    team_name = reverse_teamMapper.get(classes[i], f'Class {classes[i]}') # Use .get for safety in case a class isn't in the mapper
    plt.plot(fpr[i], tpr[i], lw=lw,
             label='ROC curve of {} (area = {:0.2f})'
             ''.format(team_name, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Multiclass')
plt.legend(loc="lower right")
plt.show()