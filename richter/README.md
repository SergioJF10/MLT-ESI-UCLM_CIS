# Richter Project
## Task 📋
For this project, we were asked to employ **supervised learning** techniques to build a model that predicts the value of a categorical label (damage grade) that represents the level of damage that the earthquake caused over a given building. This damage grade variable can have three values:
1. Level 1, low damage
2. Level 2, medium amount of damage
3. Level 3, almost complete destruction.

## Files 🗃️
This folder includes the following structure:
- `Data`: Needed for the classification process.
    - `Raw`: With the original datasets without preprocessing.
- `Models`: Final ML model for solving the problem in _joblib_ format.
- `Notebooks`: Including the main Google Colab notebook.
- `Reports`: Extra information for extra deeper explanations.
    - `Figures`: Final helpfull images for explanations.
- `Submissions`: All _CSV_ files uploaded to Driven Data platform.

## Development 💻
All these developmentes are properly documented and coded in the `Notebooks` folder.

It is also important to highlight that any preprocessing neither transformation steps over the raw data were needed.

This project was developed in four different iterations. In each iteration several algorithms were applied and/or different feature selection was employed, depending on the previous iteration's results.

### 1st Iteration
First of all, we made an initial exploratory analysis over the labels, and we clearly found an imbalance on it, being the most common the label 2 (check image below).

![Imbalance](https://github.com/SergioJF10/MLT-ESI-UCLM_CIS/blob/main/richter/Reports/Figures/Label_unbalance.png)

Then, we did an initial feature selection based on our knowledge criterion of which variables might affect considerably in the integrity of a building. Over this initial selection, NaiveBayes was applied in order to choose the best final selection.

Once this final feature selection based on Naive Bayes was made, other algorithms were tried: KNN and Decision Trees, getting an peak accuracy of $64\%$ with the Decission Trees.

### 2nd Iteration
Results from previous Decission Tree were used in order to obtain another feature selection based on the importance criteria that provides this algorithm. Over this new feature selection, the second iteration was applied.

This time, same algorithms were applied as in the previous iteration (KNN and Decision Trees) with the new variable selection, giving a peak accuracy of $70\%$ with the KNN algorithm.

### 3rd Iteration
Once we reached a considerably accurate model (the previous KNN), we now tried complex models, basically, ensembling methods.

We tried out several ensembling algorithms. We tested XgBoost algorithm and other not tree-based methods like the StackingClassifier and the VotingClassifier. Even the Adaboost technique was checked, but due to technical implementation reasons, it could not be applied. With all these algorithms, a peak accuracy of $71\%$ with the XgBoost option.

### 4th Iteration
For the final iteration, we applied two hyperparametrization techniques over the best model obtained in the previous iteration (XgBoost). The hyperparametrization techniques used in the problem were both Grid and Randomized search, being the most profitable the GridSearchCV algorithm, resulting in a $72\%$ of accuracy.

## Results 💡
The XgBoost model hyperparametrized with GridSearchCV is our final proposal for the classification task. We also analysed the evolution of the different tries we made along the whole project, as we can see in the image below.

![Accuracy](https://github.com/SergioJF10/MLT-ESI-UCLM_CIS/blob/main/richter/Reports/Figures/Acc_evolution.png)