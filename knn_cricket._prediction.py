import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import joblib

#from google.colab import drive
#drive.mount('/content/drive')
#path_to_matches="/content/drive/MyDrive/Colab Notebooks/Cricket Prediction/matches.csv"

matches=pd.read_csv('data/matches.csv')
print("Head:\n",matches.head(5))

print("Tail:\n",matches.tail(5))

matches.describe()

matches.info()

print("\nShape:",matches.shape)

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


print("value count:\n",matches.team1.value_counts())

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


winner_counts = matches['winner'].value_counts()

reverse_teamMapper = {v: k for k, v in teamMapper.items()}
winner_counts.index = winner_counts.index.map(reverse_teamMapper)

plt.figure(figsize=(12, 5)) 
sns.barplot(x=winner_counts.index, y=winner_counts.values)
plt.xticks(rotation=45, ha='right') 
plt.xlabel("Winner Team")
plt.ylabel("Number of Wins")
plt.title("Distribution of Wins per Team")
plt.tight_layout() 
plt.show()

matches[matches.isnull().any(axis=1)].head(20)

matches = matches.dropna()

print("shape:",matches.shape)

matches.head(20)

matches["winner"] = matches["winner"].round()

print("Head:\n",matches.head(20))

X = matches.drop(['winner'], axis = 1)
y = matches['winner']

#exporting cleaned data
matches.to_csv('cleaned_matches.csv', columns = ['team1','team2','season','toss_winner','winner'],index = False, sep =',')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train, y_train)

y_prediction_knn = clf_knn.predict(X_test)

#exporting predicted data
pd.DataFrame(y_prediction_knn).to_csv('predicted_matches.csv', index = False)

accuracy_score_knn = clf_knn.score(X_train, y_train)*100
print ("\n\nAccuracy of KNN Model Trainning = ",accuracy_score_knn)

accuracy_score_knn = clf_knn.score(X_test, y_test)*100
print ("\nAccuracy of KNN Model Testing = ",accuracy_score_knn)


print('\n\nRecall: ',metrics.recall_score(y_test, y_prediction_knn, zero_division=1,average='weighted'))
print('Precision:',metrics.precision_score(y_test, y_prediction_knn, zero_division=1, average='weighted'))
print('CL Report:',metrics.classification_report(y_test, y_prediction_knn, zero_division=1))

y_pred_proba = clf_knn.predict_proba(X_test)


classes = np.unique(y_train)

y_test = y_test.astype(y_train.dtype)
y_test_binarized = label_binarize(y_test, classes=classes)

fpr = dict()
tpr = dict()
roc_auc = dict()
reverse_teamMapper = {v: k for k, v in teamMapper.items()}

# Calculate ROC curve and AUC
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10,5))
lw = 2
for i in range(len(classes)):

    team_name = reverse_teamMapper.get(classes[i], f'Class {classes[i]}')
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


results = pd.DataFrame({'Actual Winner': y_test, 'Predicted Winner': y_prediction_knn})


results['Actual Winner Team'] = results['Actual Winner'].map(reverse_teamMapper)
results['Predicted Winner Team'] = results['Predicted Winner'].map(reverse_teamMapper)


correct_predictions = results[results['Actual Winner'] == results['Predicted Winner']]
incorrect_predictions = results[results['Actual Winner'] != results['Predicted Winner']]

correct_counts = correct_predictions['Actual Winner Team'].value_counts().reset_index()
correct_counts.columns = ['Team', 'Correct Predictions']

incorrect_counts = incorrect_predictions['Actual Winner Team'].value_counts().reset_index()
incorrect_counts.columns = ['Team', 'Incorrect Predictions']


prediction_summary = pd.merge(correct_counts, incorrect_counts, on='Team', how='outer').fillna(0)
prediction_summary_melted = prediction_summary.melt(id_vars='Team', var_name='Prediction Type', value_name='Count')


prediction_summary['Total'] = prediction_summary['Correct Predictions'] + prediction_summary['Incorrect Predictions']
prediction_summary = prediction_summary.sort_values('Total', ascending=False)
prediction_summary_melted['Team'] = pd.Categorical(prediction_summary_melted['Team'], categories=prediction_summary['Team'], ordered=True)


plt.figure(figsize=(15, 7))
sns.barplot(x='Team', y='Count', hue='Prediction Type', data=prediction_summary_melted)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Actual Winner Team")
plt.ylabel("Number of Predictions")
plt.title("Comparison of Actual vs. Predicted Winners")
plt.legend(title="Prediction Outcome")
plt.tight_layout()
plt.show()


cm = confusion_matrix(y_test, y_prediction_knn, labels=classes)


cm_df = pd.DataFrame(cm, index=[reverse_teamMapper.get(i, f'Class {i}') for i in classes],columns=[reverse_teamMapper.get(i, f'Class {i}') for i in classes])

plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Winner')
plt.ylabel('Actual Winner')
plt.title('Confusion Matrix of Winner Predictions')
plt.tight_layout()
plt.show()

joblib.dump(clf_knn, 'knn_model.joblib')
