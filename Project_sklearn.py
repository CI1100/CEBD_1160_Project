###
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

wine = load_wine()
columns_names = wine.feature_names

X = wine.data
y = wine.target
# What's inside this sklearn loaded dataset
print(f'keys: {wine.keys()}')
print(f'data: {wine.data}')
print(f'target: {wine.target}')
print(f'feature_names: {wine.feature_names}')

# Rebuilding pandas DF from dataset (for plotting and statistical facts)
wine_to_df = pd.DataFrame(X, columns=wine.feature_names)
wine_to_df['target'] = y

# 2.	compute and print information and summary statistics on the dataset
print ('Information:', wine_to_df.info())
print ('Summary statistics on the dataset', wine_to_df.describe())
print ('Summary statistics on the dataset', wine_to_df.shape)
print('Print first 5 rows:', wine_to_df.head())
print(wine_to_df.groupby('target').size())

# prints out how many times each value in the winery column is appearing.
os.makedirs('visualisation/1-winery_info', exist_ok=True)
#print(wine['target'].value_counts())
sns.countplot(wine_to_df['target'])
plt.savefig('visualisation/1-winery_info/winery_info_seaborn.png')
plt.clf()

# Pair plot
os.makedirs('visualisation/2-pair_plots', exist_ok=True)
columns_to_plot = list(wine_to_df.columns)
columns_to_plot.remove('target')
sns.pairplot(wine_to_df, hue='target', vars=columns_to_plot)
plt.savefig('visualisation/2-pair_plots/wine_pair_plots_seaborn.png')
plt.clf()

# 3.	compute and print correlations on the dataset
os.makedirs('visualisation/3-correlation', exist_ok=True)
print('Correlation: ', wine_to_df.corr())
matrix = np.triu(wine_to_df.corr())
sns_plot=sns.heatmap(wine_to_df.corr().round(1), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'rainbow', mask=matrix)
sns_plot.figure.savefig('visualisation/3-correlation/correlation_seaborn.png')
plt.clf()

# Splitting features and target datasets into: train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Printing original Dataset
print(f"X.shape: {X.shape}, y.shape: {y.shape}")

# Printing splitted datasets
print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

# Training a Linear Regression model with fit()

lr = LogisticRegression()
lr.fit(X_train, y_train)

# Output of the training is a model: a + b*X0 + c*X1 + d*X2 ...
print(f"Intercept per class: {lr.intercept_}\n")
print(f"Coeficients per class: {lr.coef_}\n")

print(f"Available classes: {lr.classes_}\n")
print(f"Named Coeficients for class 1: {pd.DataFrame(lr.coef_[0], columns_names)}\n")
print(f"Named Coeficients for class 2: {pd.DataFrame(lr.coef_[1], columns_names)}\n")
print(f"Named Coeficients for class 3: {pd.DataFrame(lr.coef_[2], columns_names)}\n")

print(f"Number of iterations generating model: {lr.n_iter_}")

# Predicting the results for our test dataset
predicted_values = lr.predict(X_test)

# Printing the residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')

# Printing accuracy score(mean accuracy) from 0 - 1
print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

# Printing the classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score
print('Classification Report')
print(classification_report(y_test, predicted_values))

# Printing the classification confusion matrix (diagonal is true)
print('Confusion Matrix')
print(confusion_matrix(y_test, predicted_values))

print('Overall f1-score')
print(f1_score(y_test, predicted_values, average="macro"))

#Feature importance algorithm = rank the importance of the features in the data,
# based on which feature helped the most to distinguish between the target labels.
# From this ranking we can learn which features were more and less important and select just the one's which contribute the most.

os.makedirs('visualisation/4-feature ranking', exist_ok=True)
model_feature_importance = RandomForestClassifier(n_estimators=1000).fit(X_train, y_train).feature_importances_
feature_scores = pd.DataFrame({'score':model_feature_importance}, index=list(columns_names)).sort_values('score')
print(feature_scores)
sns.barplot(feature_scores['score'], feature_scores.index)
plt.tight_layout()
plt.savefig('visualisation/4-feature ranking/wine_feature_rank_seaborn.png')
plt.clf()

# PCA or dimensionality reduction, is a technique that allows you to have a consolidated

# view of multiple features in a smaller dimension. Very useful to plot 2d and 3d visualizations
# over a large number of features
os.makedirs('visualisation/5-PCA', exist_ok=True)
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
pca = PCA(n_components=2)
proj = pca.fit_transform(wine.data)
plt.scatter(proj[:, 0], proj[:, 1], c=wine.target, edgecolors='black', label = wine.target)
axes.set_xlabel('Principal Component 1', fontsize = 15)
axes.set_ylabel('Principal Component 2', fontsize = 15)
axes.set_title('2 component PCA', fontsize = 20)
axes.legend(wine.target, loc=1, ncol=3)
plt.tight_layout()
plt.savefig(f'visualisation/5-PCA/PCA_scatter.png', dpi=300)
plt.colorbar()
plt.tight_layout()
plt.show()
plt.savefig(f'visualisation/5-PCA/PCA_colorbar.png', dpi=300)
plt.close()



