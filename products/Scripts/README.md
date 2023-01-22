# Scripts Folder
Inside this folder, we can find several **scripts used for local development** of vectorization techniques, combined with other features for preparing the input data; and also several models training and testing steps.

## Files 🗃️
The files within this folder are:
- `requirements.txt`: File with the requirements needed to execute the scripts as well as the appropriate and necessary versions.
- `tfidf_pos.py`: Script with the TF-IDF vectorization process, combined with POS tagging and undersampling to reduce the imbalances. Also, training and testing of 3 models (Naive Bayes, Decision Tree and Voting Classifier).
- `tfidf_pos_extra.py`: Script with the TF-IDF vectorization process, combined with POS tagging, extra features and undersampling to reduce the imbalances. Also, training and testing of 3 models (Naive Bayes, Decision Tree and Voting Classifier).

## Instructions 📎
For executing this scripts we recommend the following steps:

### 1. Create a venv virtual environment
If you want, you can skip this step, but we recommend it to isolate the dependencies and avoid requirements' problems.

For creating a virtual environment with the venv Python package, you can follow these instructions:
1. Install `virtualenv` Python module.
```
pip install virtualenv
```

2. Create an environment.
```
python -m venv <environment_name>
```

3. Activate the enviroment: Depending on the OS you are using.
    - Mac/Linux:
    ```
    source <environment_name>/bin/activate
    ```
    - Windows:
    ```
    .\<environment_name>\Scripts\activate.bat
    ```

### 2. Install the Scripts requirements
For this, you can just simply execute this command with this folder as the current directory.
```
pip install -r requirements.txt
```

This will automatically install all the dependencies and the correct versions of each of them.

### 3. Run the scripts
For running a script, we can just execute the following command:
```
python <script_name>.py
```

Each script will show the execution steps in the terminal and a progress bar for long tasks.
