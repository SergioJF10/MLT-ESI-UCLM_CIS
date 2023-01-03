# Submissions' Folder
In this folder we will be adding all the submissions we made along the development process of the project. We will indicate for each submission several things: (i) the **date**, (ii) the **name of the .csv** submitted, (iii) **brief explanation** about what is the content of each one; and (iv) the Driven Data **score** obtained.

- `submission_NaiveBayes` [$F1_{micro} = 0.5052$]: _Submitted on 2022-11-25 10:20:58 UTC_.
    - Includes the `damage_grade` predicted for the test sample using a Naives Bayes model, implemented with Bernoulli approach. 
- `submission_knn` [$F1_{micro} = 0.5801$]: _Submitted on 2022-12-02 13:22:11 UTC_.
    - Includes the `damage_grade` predicted for the test sample using a kNN model, implemented with normalized data, a feature selection of 19 variables, and a total of $k=128$ neighbors. 
- `submission_knn_iter2` [$F1_{micro} = 0.6938$]: _Submitted on 2022-12-03 10:12:01 UTC_.
    - Includes the `damage_grade` predicted for the test sample using a kNN model, implemented with normalized data, a feature selection of 7 variables, and a total of $k=16$ neighbors.
- `submission_knn_uniform` [$F1_{micro} = 0.7040$]: _Submitted on 2022-12-06 09:45:54 UTC_.
    - Includes the `damage_grade` predicted for the test sample using a kNN model, implemented with normalized data, a feature selection of 7 variables, using uniform weight and a total of $k=16$ neighbors.
- `submission_dt` [$F1_{micro} = 0.6482$]: _Submitted on 2022-12-03 15:56:09 UTC_.
    - Includes the `damage_grade` predicted for the test sample using a DT model, with all the features, and $depth=25$. 
- `submission_dt_iter2` [$F1_{micro} = 0.6486$]: _Submitted on 2022-12-03 15:50:25 UTC_.
    - Includes the `damage_grade` predicted for the test sample using a DT model, a feature selection of 7 variables, and $depth=23$. 
- `submission_xb_rscv` [$F1_{micro} = 0.6987$]: _Submitted on 2022-12-09 10:02:29 UTC_.
    - Includes the `damage_grade` predicted for the test sample using XgBoost ensembling algorithm and hyperparametrization optimized by means of the RandomizedSearch cross validation algorithm.
- `submission_stk` [$F1_{micro} = 0.7166$]: _Submitted on 2022-12-09 12:56:18 UTC_.
    - Includes the `damage_grade` predicted for the test sample using StackingClassifier with kNN (uniform) and Decision Trees (iter2) as weak learners and as meta learner a Logistic Regression.
- `submission_xgboost` [$F1_{micro} = 0.6990$]: _Submitted on 2022-12-10 13:02:54 UTC_.
    - Includes the `damage_grade` predicted for the test sample using the ensemble model based on decision trees XgBoost.
- `submission_xg_gscv` [$F1_{micro} = 0.7213$]: _Submitted on 2022-12-09 14:41:45 UTC_.
    - Includes the `damage_grade` predicted for the test sample using the ensemble model based on decision trees XgBoost using GridSearchCV hyperparametrization algorithm.     
- `submission_vote` [$F1_{micro} = 0.7144$]: _Submitted on 2022-12-15 00:58:53 UTC_.
    - Includes the `damage_grade` predicted for the test sample using VotingClassifier with kNN (uniform), Decision Trees (iter2) and StackingClassifier as baseline models using hard vote.
