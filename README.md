Activity 2: Binary Classification using Logistic Regression
Course: Regression and Classification Algorithms (RCA)
Department: Artificial Intelligence and Data Science (AI & DS)
Class: T.Y. B.Tech | Semester: V | Academic Year:
2026--27
Assessment: CIE 2 -- Activity 2 (5 Marks)
1. Activity overview
This activity builds a Logistic Regression model for a two-class
classification problem. The model estimates class probabilities and
converts them into class labels using a decision threshold.

The notebook includes dataset preparation, stratified train-test
splitting, feature scaling, Logistic Regression training, probability
prediction, confusion-matrix analysis, classification metrics,
comparison with a majority-class baseline, threshold analysis at 0.3,
0.5 and 0.7, ROC curve analysis, and prediction for a new example.
2. Dataset details
- Dataset name: Breast Cancer Wisconsin Diagnostic Dataset
- Source: Scikit-learn
- Official dataset documentation:
  https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset
- Number of observations: 569
- Number of input features: 30 numerical features
- Target column: target
- Class 0: Malignant
- Class 1: Benign
The dataset contains numerical measurements computed from digitized
images of fine needle aspirates (FNA) of breast masses. The target has
exactly two classes.

Important: This dataset and notebook are for educational
classification practice only. The model is not clinically validated and
must not be used for medical diagnosis, screening, or treatment
decisions.
3. Repository contents
  File                                                          Description
  Activity2_Logistic_Regression_Binary_Classification.ipynb   Jupyter Notebook containing the
code, explanations, metrics,
visualizations and predictions
README.md                                                   Project overview, workflow, results
and execution instructions
The notebook loads the dataset directly using Scikit-learn's
load_breast_cancer(as_frame=True) function. A separate CSV dataset
file is not required to run it.
4. Tools and libraries used
- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook or Google Colab
5. Implementation workflow
Step 1: Import libraries
Import the required libraries for data manipulation, visualization,
preprocessing, Logistic Regression, baseline modelling and evaluation
metrics. The notebook also displays the Python and library versions.
Step 2: Load and understand the dataset
Load the Breast Cancer Wisconsin Diagnostic dataset using Scikit-learn.
Display the first records, dataset shape, feature count, target class
names, data types and dataset information.
Step 3: Inspect and clean the data
Check missing values, duplicate records, summary statistics and class
counts. Remove exact duplicate rows if present and fill missing
numerical feature values using their medians if needed.
Step 4: Visualize class distribution
Plot the number of malignant and benign records and display the class
proportions.
Step 5: Prepare data and stratified split
Separate the input features (X) and target (y). Split the data into
80% training and 20% testing sets using stratify=y and
random_state=42, preserving approximately the same class proportions
in both sets.
Step 6: Understand the sigmoid function
Plot the sigmoid function, which converts a model score into a
probability between 0 and 1. At the default threshold of 0.5, a record
is classified as class 1 when its predicted probability is at least 0.5;
otherwise, it is class 0.
Step 7: Train Logistic Regression
Create a Scikit-learn Pipeline containing StandardScaler and
LogisticRegression(max_iter=5000, random_state=42). Fit the pipeline
using training data only. Scaling inside the pipeline prevents the
scaler from learning information from the test set.
Step 8: Interpret coefficients
Display the model intercept and feature coefficients. Since class 1 is
benign, a positive coefficient increases the log-odds of the benign
class for a one-standard-deviation increase in that feature, holding
other features constant. Coefficients show associations, not causal
effects.
Step 9: Generate probabilities and class labels
Use predict_proba() to obtain the probability of class 1 (benign).
Convert probabilities to predicted labels using the default threshold of
0.5 and compare sample predictions with actual labels.
Step 10: Confusion matrix and classification metrics
Generate a confusion matrix and calculate: - Accuracy: proportion of
all predictions that are correct. - Precision: proportion of
predicted benign records that are actually benign. - Recall /
sensitivity: proportion of actual benign records correctly predicted
as benign. - Specificity: proportion of actual malignant records
correctly predicted as malignant. - F1-score: harmonic mean of
precision and recall. - ROC-AUC: ability of the model to rank
positive-class records above negative-class records across thresholds.

For this notebook, the positive class is benign (1). Therefore: -
TN: actual malignant predicted malignant. - FP: actual malignant
predicted benign. - FN: actual benign predicted malignant. - TP:
actual benign predicted benign.
Step 11: Classification report
Display precision, recall, F1-score and support for both malignant and
benign classes.
Step 12: Compare with majority-class baseline
Train a DummyClassifier(strategy="most_frequent") on the training
data. Compare its test accuracy and positive-class precision, recall and
F1-score with Logistic Regression.
Step 13: Compare decision thresholds
Evaluate thresholds 0.3, 0.5 and 0.7 using the same predicted
probabilities. For each threshold, calculate the number of predicted
positive (benign) records, accuracy, precision, recall, specificity and
F1-score.

A lower threshold predicts more records as benign and can increase
benign recall, while a higher threshold predicts fewer records as
benign. Threshold selection depends on the relative cost of false
positives and false negatives.
Step 14: ROC curve and ROC-AUC
Plot the ROC curve using the test-set probabilities and calculate
ROC-AUC. The ROC curve displays the trade-off between true positive rate
(recall) and false positive rate across possible thresholds.
Step 15: Visualize predicted probabilities
Plot predicted probabilities of the benign class against the feature
with the strongest absolute training-set correlation with the target.
The notebook identifies this feature from the training data.
Step 16: Predict a new example
Use the first record from the held-out test set as a new example.
Display its predicted probability of being benign and its predicted
class at threshold 0.5.
Step 17: Final summary
Print the majority-class baseline accuracy, Logistic Regression metrics,
ROC-AUC and threshold comparison table.
6. Results from the notebook
The following values were obtained by executing the notebook with its
stated code, including random_state=42 and the 80:20 stratified split.
Logistic Regression at threshold 0.5
  Metric                                          Result
  Accuracy                               0.9825 (98.25%)
Precision (positive = benign)          0.9861 (98.61%)
Recall (positive = benign)             0.9861 (98.61%)
Specificity (negative = malignant)     0.9762 (97.62%)
F1-score (positive = benign)           0.9861 (98.61%)
ROC-AUC                                0.9954 (99.54%)
Confusion matrix at threshold 0.5
Rows represent actual classes and columns represent predicted classes,
ordered as malignant (0), benign (1):

  Actual  Predicted     Malignant (0)   Benign (1)
  Malignant (0)                    41            1
Benign (1)                        1           71

Thus, TN = 41, FP = 1, FN = 1 and TP = 71.
Majority-class baseline
The majority-class baseline achieved 63.16% accuracy, compared with
98.25% accuracy for Logistic Regression on the same test set.
Decision-threshold comparison
Threshold   Accuracy   Precision    Recall   Specificity   F1-score

      0.3     98.25%      97.30%   100.00%        95.24%     98.63%
      0.5     98.25%      98.61%    98.61%        97.62%     98.61%
      0.7     94.74%      98.53%    93.06%        97.62%     95.71%
Threshold decision
For this coursework comparison, threshold 0.5 is a reasonable
balanced choice: it gives high precision, recall, F1-score and
specificity. Threshold 0.3 gives 100% recall for the benign class in
this test split, but it also produces more false positives (malignant
records predicted as benign). Threshold 0.7 reduces benign recall and
overall accuracy. Threshold selection in real medical use requires
clinical validation and expert assessment of error costs; these results
do not justify clinical use.
New example prediction
The notebook uses the first held-out test record as the new example. In
the executed run, the predicted probability of benign was approximately
0.0000, and the predicted class was Malignant (0).
7. How to run the notebook
Using Google Colab
1. Download the notebook from this repository.
2. Open https://colab.research.google.com/.
3. Select File → Upload notebook.
4. Upload Activity2_Logistic_Regression_Binary_Classification.ipynb.
5. Select Runtime → Run all.
6. Wait for the code cells, tables and plots to finish executing.
7. Save the executed notebook with its outputs.
Running locally
Install the required libraries:
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
Start Jupyter Notebook:
jupyter notebook
Open the notebook and run all cells from top to bottom.
8. Conclusion
The Breast Cancer Wisconsin Diagnostic dataset was used to build a
binary Logistic Regression model. The dataset was inspected and cleaned,
then divided using an 80:20 stratified train-test split. StandardScaler
and Logistic Regression were combined in a Pipeline to ensure that
scaling was fitted only on training data.

At threshold 0.5, the model achieved 98.25% accuracy, 98.61% precision,
98.61% recall, 98.61% F1-score and 99.54% ROC-AUC, outperforming the
majority-class baseline accuracy of 63.16%. Comparing thresholds 0.3,
0.5 and 0.7 demonstrated how the decision threshold changes precision,
recall, specificity and the types of classification errors.

These results are specific to this dataset and test split. The model is
intended only for educational purposes and is not suitable for medical
diagnosis or decision-making.
9. Limitations
- The dataset is an educational dataset and may not represent all
  real-world patient populations.
- The model's test performance is based on one stratified train-test
  split and may vary with another split.
- Precision, recall and other metrics depend on the chosen positive
  class and decision threshold.
- False positives and false negatives have different meanings in this
  notebook because class 1 is benign and class 0 is malignant.
- High test metrics do not establish clinical validity or guarantee
  performance on new populations.
- This notebook must not be used for diagnosis, screening or treatment
  decisions.
10. References
1. Scikit-learn -- Breast Cancer Wisconsin dataset:
   https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset
2. Scikit-learn -- load_breast_cancer documentation:
   https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
3. Scikit-learn -- Logistic Regression:
   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
4. Scikit-learn -- DummyClassifier:
   https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
