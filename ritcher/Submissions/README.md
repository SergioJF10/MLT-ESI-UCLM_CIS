# Submissions' Folder
In this folder we will be adding all the submissions we made along the development process of the project. We will indicate for each submission several things: (i) the **date**, (ii) the **name of the .csv** submitted, (iii) **brief explanation** about what is the content of each one; and (iv) the Driven Data **score** obtained.

- `submission_NaiveBayes` [$F1_{micro} = 0.5052$]: _Submitted on 2022-11-25 10:20:58 UTC_.
    - Includes the `damage_grade` predicted for the test sample using a Naives Bayes model, implemented with Bernoulli approach. 
- `submission_knn` [$F1_{micro} = 0.5801$]: _Submitted on 2022-12-02 13:22:11 UTC_.
    - Includes the `damage_grade` predicted for the test sample using a kNN model, implemented with normalized data, a feature selection of 19 variables, and a total of $k=128$ neighbors. 
- `submission_knn_iter2` [$F1_{micro} = 0.6938$]: _Submitted on 2022-12-03 10:12:01 UTC_.
    - Includes the `damage_grade` predicted for the test sample using a kNN model, implemented with normalized data, a feature selection of 7 variables, and a total of $k=16$ neighbors.
- `submission_dt` [$F1_{micro} = 0.6482$]: _Submitted on 2022-12-03 15:56:09 UTC_.
    - Includes the `damage_grade` predicted for the test sample using a decision tree model, with all the features, and $max_depth=25$. 
- `submission_dt` [$F1_{micro} = 0.6486$]: _Submitted on 2022-12-03 15:50:25 UTC_.
    - Includes the `damage_grade` predicted for the test sample using a decision tree model, a feature selection of 7 variables, and $max_depth=23$. 
