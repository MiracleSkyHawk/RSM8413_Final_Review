#  Packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("df.txt")

# ------------------------------------------------
####   Bar chart  ####

# seaborn bar chart
ax = sns.countplot(df["Churn?"])
ax.set_title("Churning Customers")

# matplotlib bar chart
df["Churn?"].value_counts().plot(kind='bar', title= "Churning Customers")

# ------------------------------------------------
####  Stacked Bar chart and Crosstab  ####


# pandas crosstab
churn_crosstab = pd.crosstab(df["Churn?"], df["Int'l Plan"], margins=False)

# seaborn stacked bar charts
sns.countplot(x="Int'l Plan", hue="Churn?", data=df)

churn_crosstab = churn_crosstab.transpose()
churn_crosstab.plot(kind = 'bar', stacked = True)

# stacked bar chart normalized
churn_crosstab_norm = churn_crosstab.div(churn_crosstab.sum(axis=1),axis=0)
churn_crosstab_norm.plot(kind = 'bar', stacked=True)

# ------------------------------------------------
####  Histograms  ####

# pandas histogram
df.hist(figsize=(14,10))
df.plot(figsize=(14,10), kind='density', subplots=True, layout=(4,4), sharex=False)

# seaborn histogram
sns.kdeplot(df["CustServ Calls"])

# matplotlib histogram
churn_csc_T = df[df["Churn?"] == "True."]["CustServ Calls"]
churn_csc_F = df[df["Churn?"] == "False."]["CustServ Calls"]

plt.hist([churn_csc_T, churn_csc_F], bins = 10, stacked = True)
plt.legend(['Churn = True', 'Churn = False'])
plt.title('Histogram of Customer Service Calls with Churn Overlay')
plt.xlabel('Customer Service Calls')
plt.ylabel('Frequency')
xlabels = np.arange(10)  # the labels
xpos = [x*0.9+0.45 for x in xlabels]  # the label locations
plt.xticks(xpos, xlabels)

# matplotlib normalized histogram
(n, bins, patches) = plt.hist([churn_csc_T, churn_csc_F], bins = 10, stacked = True)
n[1] = n[1] - n[0]
n_table = np.column_stack((n[0], n[1]))
n_norm = n_table / n_table.sum(axis=1)[:, None]
ourbins = np.column_stack((bins[0:10], bins[1:11]))

plt.bar(x = ourbins[:,0], height = n_norm[:,0], width = ourbins[:, 1] - ourbins[:, 0])
plt.bar(x = ourbins[:,0], height = n_norm[:,1], width = ourbins[:, 1] - ourbins[:, 0], bottom = n_norm[:,0])

plt.legend(['Churn = True', 'Churn = False'])
plt.title('Normalized Histogram of Customer Service Calls with Churn Overlay')
plt.xlabel('Customer Service Calls')
plt.ylabel('Proportion')
xpos = [x-0.45 for x in xpos]
plt.xticks(xpos, xlabels)

# ------------------------------------------------
####  Box Plots  ####

# pandas boxplot
df.boxplot(figsize=(14,10))
plt.xticks(rotation=90)

# matplotlib individual boxplot
df.plot(figsize=(14,10), kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)

# seaborn boxplot
sns.boxplot(x = df["Churn?"], y = df["CustServ Calls"], data = df)

# ------------------------------------------------
####  Scatter Plots  ####

# seaborn scatter plot
sns.scatterplot(x = "Day Mins", y = "Eve Mins", hue="Churn?", data = df)


# seaborn scatter matrix
from pandas.plotting import scatter_matrix

scatter_matrix(df)

# matplotlib scatter plot
for column in df.columns:
    if column != 'Median_income':
        df.plot(kind = "scatter", x = 'Median_income', y = column, alpha=0.1)

# ------------------------------------------------
####  Correlation Plots  ####

# pandas correlation matrix
correlations = df.corr()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)

ticks = range(0,16,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
numeric_columns = df.select_dtypes(include='number')
ax.set_xticklabels(numeric_columns, rotation=90)
ax.set_yticklabels(numeric_columns)


# seaborn correlation heatmap
sns.heatmap(df.corr(method='pearson'))

# seaborn correlation heatmap
plt.figure(figsize=(25, 15))
plt.suptitle('Correlations', fontsize = 30, color= 'teal')
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.show()

# ------------------------------------------------
####  KMeans Plots  ####

# visualize silhouette_sample_score
fig, ax = plt.subplots(7, 2, figsize=(10,30))
for k in range(2,15):
    
    np.random.seed(84) 

    km = KMeans(n_clusters=k)
    q, mod = divmod(k, 2)

    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(scaled_df) 

# visualize inertias
ks = range(1,20)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    
    np.random.seed(seed)
    
    model = KMeans(n_clusters = k)
    
    # Fit model to samples
    model.fit(scaled_df)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# visualize silhouette line plot
ks = range(2,15)
silhouette_score_list = []
for k in ks:
    
    np.random.seed(seed)
    
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)
    
    # Fit model to samples
    model.fit_predict(scaled_df)
    
    # Append the inertia to the list of inertias
    s_score = silhouette_score(scaled_df, model.labels_, metric='euclidean')
    silhouette_score_list.append(model.inertia_)
    
    
plt.plot(ks, silhouette_score_list, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('silhouette score')
plt.xticks(ks)
plt.show()