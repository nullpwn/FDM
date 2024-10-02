Installation:

1. Ensure that you have Python installed on your system (Python 3 recommended).
2. Install the required libraries by running: pip install pandas numpy scikit-learn

Execution:

1. Place the 'task_code.ipynb' file in a directory.
2. Ensure that the 'Images.csv' and 'EdgeHistogram.csv' files are in the same directory as the notebook.
3. Open a terminal or command prompt in the directory containing the files.
4. Launch Jupyter Notebook: jupyter notebook
5. Open the 'pr02.ipynb' notebook in the Jupyter Notebook interface.
6. Run each cell sequentially to execute the code.

Operation:

1. The program reads the 'images.csv' and 'EdgeHistogram.csv' files to create a merged DataFrame.
2. It performs a train-test split on the data.
3. Three models (RandomForest, SVM, KNN) are trained and evaluated using GridSearchCV for hyperparameter tuning.
4. Accuracy scores for each model are printed.
5. Confusion matrices are generated and saved in CSV files ('group022_result1.csv', 'group022_result2.csv', 'group022_result3.csv').
6. Best hyperparameters for each model are saved in CSV files ('group022_parameters1.csv', 'group022_parameters2.csv', 'group022_parameters3.csv').