# Worldcup Project
## Task 📋
This tasks was related with the concepts of **unsupervised learning** techniques, corresponding with the first part of the course. Regarding the deriverable, the problem was divided in three main parts:
1. Identification of **outliers** in the dataset.
2. Use of several **clustering algorithms** to identify groups and characterize them.
3. Try to optimize the **feature selection** by means of clustering algorithms too.

## Files 🗃️
This folder includes the following structure:
- `Data`: Needed for the clustering process.
    - `Raw`: With the original _worldcup_2018_final_data.csv_ dataset.
- `Models`: Final clustering models in _joblib_ format.
- `Notebooks`: All the relevant Google Colab Notebooks.
- `References`: Documents used as bases for some decissions.
- `Reports`: Extra information for further explanations.
    - `Figures`: Final helpfull images for explanations.

## Development 💻
All these developmentes are properly documented and coded in the `Notebooks` folder.

It is also important to highlight that any preprocessing neither transformation steps over the raw data were needed.
### Feature Selection
For optimizing the feature selection, we first tried a **domain knowledge** approach based on the attacking features provided in the dataset feature groups in `References/Dataset Features`. 

Then, after a first filtering, we started the process of clustering. Initially, with a quick variable analysis and a correlation analysis to be taken into account in further steps. Secondly, we used a Principal Component Analysis (PCA) algorithm to try to estimate the appropriate variance ratio as well as getting a way for visualizating the features graphically.

Finally, considerating the results of a hierarchical clustering over the filtered and preprocessed features, we tried to extract the final list of features.

### Clustering
First of all, we initially applied PCA for dimensionality reduction. For this step, we also tried several normalization techniques (i.e., MinMaxScaler, MaxAbsScaler and StandardScaler).

Then, we applied DBSCAN algorithm to detect outliers on the data, with two different distance measures (euclidean and manhattan metrics). This process ended up on 6 removed outliers.

Then, two algorithms were applied for clustering: Kmeans and Hierarchical clustering. The results of these two algorithms are explained later.

## Results 💡
Regarding the results of this study, we came up with two different approaches for solving the problem, depending on which algorithm was used for the clustering task.

### KMeans results
As we can see in the figure below, five different groups were found:
- Group 0: Good deadball and cross game plus **adaptative capability**.
- Group 1: **Disappointing** performance.
- Group 2: **Center-oriented and long shots** game style.
- Group 3: Good **deadball and cross** game.
- Group 4: **Bad performance**.

![Figure 1](https://github.com/SergioJF10/MLT-ESI-UCLM_CIS/blob/main/worldcup/Reports/Figures/Kmeans_results.png)

### Hierarchical results
Again, as we can see in the figure below, for this occasion, three groups were found:
- Group 1: **Bad performance**.
- Group 2: **Deep game** style.
- Group 3: **Long shots game** style.

![Figure 2](https://github.com/SergioJF10/MLT-ESI-UCLM_CIS/blob/main/worldcup/Reports/Figures/Hierarchical_results.png)