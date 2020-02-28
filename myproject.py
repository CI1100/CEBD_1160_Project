#---------HOMEWORK_5--------------------------------------
# 1.load and organize the data in a pandas data frame format,
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma

# please notice that some datasets will require that you manually fix the headers as they're not present in the dataset,
# or adapt the separators used (spaces, commas, etc).
##	Tip: For the boston dataset, your separator will be `sep='\\s+'`. This will mitigate the problem with multiple spaces!
##	Tip: In order to fix the column names, you might have to manually assign the values to it as in the examples!
wine_df = pd.read_csv('wine.data', sep =',', header=0)

# 2.	compute and print information and summary statistics on the dataset
print ('Information:', wine_df.info())
print ('Summary statistics on the dataset', wine_df.describe())
print ('Summary statistics on the dataset', wine_df.shape)
print('Print first 5 rows:', wine_df.head())
print(wine_df.groupby('Winery').size())

# 3.	compute and print correlations on the dataset
print('Correlation: ', wine_df.corr())
matrix = np.triu(wine_df.corr())
plt.subplots(figsize=(10,10))
sns_plot=sns.heatmap(wine_df.corr().round(1), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'rainbow', mask=matrix)
plt.subplots_adjust(bottom=0.28, left=0.28)
sns_plot.figure.savefig('visualisation/correlation_seaborn.png')
plt.clf()
# 4.	if the question you had previously formulated was not a predictive question,
# you should be able to answer it using pandas,
# (if you didn’t formulate a question, use pandas to extract an insight from the dataset you’re working on)
#--------------------Homework_6-----------------------------------------------------------------------
#1.	Using your project dataset, generate the items below:
#a.	At least one line plot
plt.plot(wine_df['Alcohol'], color='red')
plt.title('Alcohol')
plt.ylabel('Alcohol')
plt.savefig(f'visualisation/alcohol_plot.png', format='png')
plt.clf()

#b.	At least one histogram plot
plt.hist(wine_df['Malic acid'], bins=3, color='g')
plt.title('Malic acid')
plt.xlabel('Malic acid')
plt.ylabel('Count')
plt.savefig(f'visualisation/Malic_acid_hist.png', format='png')
plt.clf()

#c.	At least one scatter plot
plt.style.use("ggplot")
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.scatter(wine_df['Color intensity'], wine_df['Alcohol'], alpha=0.7, label='Alcohol')
axes.scatter(wine_df['Color intensity'], wine_df['Ash'], alpha=0.7, label='Ash')
axes.scatter(wine_df['Color intensity'], wine_df['Flavanoids'], alpha=0.7, label='Flavanoids')
axes.set_xlabel('Color intensity')
axes.set_ylabel('Alcohol/Ash / Flavanoids')
axes.set_title(f'Color intensity comparisons')
axes.legend()
plt.savefig(f'visualisation/multiplot_scatter.png', dpi=300)
plt.clf()

plt.close()
#2.	Submit the code and generated images in your visualization-homework repository.
