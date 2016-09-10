
import pandas as pd 
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
import numpy as np
import sklearn as sk 
import psycopg2 as psy
import sqlalchemy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
import seaborn as sns
from __future__ import print_function
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.tools import FigureFactory as FF
tls.set_credentials_file(username='lasisioo', api_key='qwwplqpgcq')



df = pd.read_csv("../assets/airportdata.csv")
df = df.drop(["Unnamed: 0"], axis=1)
#df.columns


toint = ["year", "departure cancellations", "arrival cancellations", "departure diversions", "arrival diversions"]
df[toint] = df[toint].astype(int)
df.head()


df.dtypes


df.shape


sns.heatmap(df.corr())


df["airport"] = df["airport"].astype("category")
cat_columns = df.select_dtypes(['category']).columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)


normalized = ["airport", "year", "departure cancellations", "arrival cancellations", 
              "departure diversions", "arrival diversions", "percent on-time gate departures", 
              "percent on-time airport departures", "percent on-time gate arrivals", 
              "average_gate_departure_delay", "average_taxi_out_time", "average taxi out delay", 
              "average airport departure delay", "average airborne delay", "average taxi in delay", 
              "average block delay", "average gate arrival delay"]
df[normalized] = df[normalized].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(15, 10))
plt.suptitle("Departure Cancellations and Airport Operations", size=16)

sns.set_style('whitegrid')
fig1 = sns.regplot(x="average_gate_departure_delay", y="departure cancellations", data=df, ax=ax1)
ax1.set_xlabel("Average Delay for Gate Departure",fontsize=10)
ax1.set_ylabel("Departure Cancellations",fontsize=10)

fig2 = sns.regplot(x="percent on-time airport departures", y="departure cancellations", data=df, ax=ax2)
ax2.set_xlabel("Percentage of On-Time Airport Departured",fontsize=10)
ax2.set_ylabel("Departure Cancellations",fontsize=10)

fig3 = sns.regplot(x="percent on-time gate departures", y="departure cancellations", data=df, ax=ax3)
ax3.set_xlabel("Percentage of On-Time Gate Departures",fontsize=10)
ax3.set_ylabel("Departure Cancellations",fontsize=10)

fig4 = sns.regplot(x="average taxi out delay", y="departure cancellations", data=df, ax=ax4)
ax4.set_xlabel("Average Taxi Out Delay",fontsize=10)
ax4.set_ylabel("Departure Cancellations",fontsize=10)

fig5 = sns.regplot(x="average airport departure delay", y="departure cancellations", data=df, ax=ax5)
ax5.set_xlabel("Average Airport Departure Delay",fontsize=10)
ax5.set_ylabel("Departure Cancellations",fontsize=10)

fig6 = sns.regplot(x="departure diversions", y="departure cancellations", data=df, ax=ax6)
ax6.set_xlabel("Departure Diversions",fontsize=10)
ax6.set_ylabel("Departure Cancellations",fontsize=10)


fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(15, 15))
plt.suptitle("Cancellations and Airport Operations", size=16)

sns.set_style('whitegrid')
fig1 = sns.regplot(x="arrival diversions", y="arrival cancellations", data=df, ax=ax1)
ax1.set_xlabel("Average Arrival Diversions",fontsize=10)
ax1.set_ylabel("Arrival Cancellations",fontsize=10)

fig2 = sns.regplot(x="percent on-time gate arrivals", y="arrival cancellations", data=df, ax=ax2)
ax2.set_xlabel("Percent On-Time Gate Arrivals",fontsize=10)
ax2.set_ylabel("Arrival Cancellations",fontsize=10)

fig3 = sns.regplot(x="average taxi in delay", y="arrival cancellations", data=df, ax=ax3)
ax3.set_xlabel("Average Taxi in Delay",fontsize=10)
ax3.set_ylabel("Arrival Cancellations",fontsize=10)

fig4 = sns.regplot(x="average gate arrival delay", y="arrival cancellations", data=df, ax=ax4)
ax4.set_xlabel("Average Gate Arrival Delay",fontsize=10)
ax4.set_ylabel("Arrival Cancellations",fontsize=10)


df.dropna(axis=0, inplace=True)


df.count()


sns.heatmap(df.corr())


X = df[["percent on-time airport departures", 
        "percent on-time gate arrivals", "average_gate_departure_delay", 
        "average_taxi_out_time", "average taxi out delay", 
        "average airport departure delay", "average airborne delay", 
        "average taxi in delay", "average block delay", 
        "average gate arrival delay"
       ]]


range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
   
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhoutte score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X.ix[:,0], X.ix[:,1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:,0], centers[:,1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette Analysis for KMeans Clustering on Sample Data "
                  "with %d Clusters" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()




XStd = StandardScaler().fit_transform(X)
cov_mat = np.cov(XStd.T)
eigenValues, eigenVectors = np.linalg.eig(cov_mat)
eigenSum = sum(eigenValues)
expVar = [(i / eigenSum)*100 for i in sorted(eigenValues, reverse=True)]
cumExpVar = np.cumsum(expVar)
print (expVar, cumExpVar)


#Bar chart for explained variance
trace = Bar(
        x=["PC %s" %i for i in range(1,12)], 
        y=expVar,
        name = "Explained Variance"
        )

trace2 = Scatter(
        x=["PC %s" %i for i in range(1,12)], 
        y=cumExpVar,
        name= "Cumulative Explained Variance"
        )

data = Data([trace, trace2])

layout = Layout(
        yaxis = YAxis(title = "Explained Variance Ratio"),
        title = "Explained Variance for each Principal Component"
        )
        

fig = Figure(data=data, layout=layout)
py.iplot(fig)


pca = PCA(n_components=3)
pcaDB = pd.DataFrame(pca.fit_transform(XStd), columns = ["PC1", "PC2", "PC3"])


# Correlation between PCA and features
pcaFeatureCorr = pd.DataFrame(pca.components_, columns=X.columns, index = [["PC1", "PC2", "PC3"]])
pcaFeatureCorr



# 3-D graph of PC1, PC2, and PC3
x, y, z = pcaDB["PC1"], pcaDB["PC2"],  pcaDB["PC3"]

trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-colorscale')
