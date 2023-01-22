# Products Project
## Task 📋
This task was related with Natural Language Processing (NLP) concepts and techniques, corresponding with the laboratory part of this subject. Regarding the deriverable, the problem was divided in these main parts:

1. Preprocessing.
2. Vectorization.
3. Feature Selection.
4. Classification Algorithms.

The main goal is to obtain several models that, given an text with an opinion for a product, the model predicts the score that user will give to the product (from 1 to 5).

## Files 🗃️
This folder includes the following structure:
- `Data`: Needed for the NLP process.
    - `Raw`: With the original _products.csv_ dataset.
    - `Interim`: With json files with data preprocessed and transformed according to NLP techniques.
- `Models`: With the best model from all the obtained ones.
- `Notebooks`: All the relevant Google Colab Notebooks.
    - `Models`: Colab Notebooks with the Vectorization and Model obtaining steps, for 4 different approaches.
    - `NLP`: Colab Notebook with the preprocessing and transformation steps according to NLP techniques and concepts.
- `Reports`: Extra information for firther explanations.
    - `Figures`: Final helpfull images for explanations.
- `Scripts`: With several scripts for local executions of some vectorization and model obtaining steps. Including also the _requirements.txt_ file (**instructions inside** the folder).

## Disclaimer 🚩
For some developments, we could not execute all the models and techniques in the corresponding Google Colab notebook due to RAM shortage reasons in standard accounts.

For mitigating this, we took two decisions in general:
1. Place checkboxes for choosing whether to execute the notebook with or without heavy memory steps.
2. Develop some algorithms and models locally, by means of Python [Scripts](https://github.com/SergioJF10/MLT-ESI-UCLM_CIS/tree/main/products/Scripts) and write the code also in the notebook.

The scripts can be found in the folder `Scripts` in this project directory.

## Development 💻
All these developmentes are properly documented and coded in the `Notebooks` folder.

Mainly we have two parts for this project: preprocessing/transformation and vectorization/model obtaining.

### Preprocessing
This process is explained and developed in the notebook in [`Notebooks/NLP/NLP_products.ipynb`](https://github.com/SergioJF10/MLT-ESI-UCLM_CIS/blob/main/products/Notebooks/NLP/NLP_products.ipynb).

For the preprocessing, first of all, we needed to load the file and obtain the relevant data from it. This was not a crutial step, since there were a lot of _typos_ in the _csv_ format that they made it impossible. This process involves removing wrong characters and joining desired columns.

Once the file was correctly loaded, we performed the preprocessing, by removing several characters and chaging the necessary letters. In more details, the process was:
1. Remove useless characters.
2. Convert capital letters to lower case.
3. Useless tokens, such as HTML tags.
4. Expand contractions.
5. Remove emojis.
6. Remove numbers.

Then, we obtained several features data, to be used in further vectorization and model training.

![Worcloud](https://github.com/SergioJF10/MLT-ESI-UCLM_CIS/blob/main/products/Reports/Figures/Wordcloud.png)

After that, we removed stopwords and lemmatize them, obtaining the worlcloud shown above.

### Vectorization & Model Obtaining
Once we have the data ready to be used, we applied four approaches:
1. **TF-IDF vectorization**: Including the vectorization itself, as well as undersampling techniques to reduce the impact of the imbalance in labels. Three models are applied:
    - Naive Bayes (Multinomial Approach).
    - Decision Tree.
    - Voting Classifier.

2. **TF-IDF and N-Grams**: Including the vectorization and n-gram generation in a range from uni-grams to tri-grams. Same as before, undersampling techniques are placed (taking Disclaimer info into account). Same three models were applied.

3. **TF-IDF and POS tagging**: Includes the vectorization and POS tagging info (number of adjectives in the documents). As usual, we also indicate the undersampling process (taking into account the Disclaimer) and same three models.

    We did not include the N-grams in this neither in the fourth approach due to the extreme huge memory demanding of the n-gram generating task.

4. **TF-IDF, POS tagging and Extra Features**: Includes the vectorization, POS tagging with the number of adjectives and extra features. These extra features are the number of words and the number of sentences in a document. Same as before, undersampling is developed and the same three models are included.

## Results 💡
For evaluating the final results, we used three metrics (as we were asked in the task description):

1. F1 Score 
2. Precision
3. Recall

All of them ere used with tuned `average` parameter for multiclass classification.

According to those metrics, our best model is obtained with the first approach and a Naive Bayes model, with a $64\%$ of accuracy in the F1 Score. This model is saved as a _joblib_ in [`Models/mngb_model.joblib`](https://github.com/SergioJF10/MLT-ESI-UCLM_CIS/blob/main/products/Models/mngb_model.joblib).