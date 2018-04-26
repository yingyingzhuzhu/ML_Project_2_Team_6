# MLProject2
This project involves Classification using the following algorithms:
1. Support Vector Machines
2. Relevance Vector Machines
3. Gaussian Process Regression

Prerequisites:
1. scipy.io
2. numpy
3. sklearn.model_selection -> KFold
4. sklearn.model_selection ->  train_test_split
5. sklearn.gaussian_process ->  GaussianProcessClassifier
6. skrvm ->  RVC
7. sklearn.svm -> SVC
8. sklearn.gaussian_process.kernels ->  RBF
9. time ->  time
10. sklearn.decomposition ->  PCA
11. sklearn.metrics ->  classification_report
12. sklearn.metrics ->  confusion_matrix
13. sklearn.metrics -> accuracy_score
14. pandas -> pd


Following are the methods:
1. def TrainMyClassifier(XEstimate, YEstimate, XValidate, TrainMyClassifierParameters)
Input Parameters:
    XEstimate: This is the input data of the training set.
    YEstimate: This is the labels of the XEstimate(1-5 in our case)
    XValidate: This is the input data of the validation set
    TrainMyClassifierParameters: This is an array.
    Example:
      1. For SVM:
          TrainMyClassifierParameters[0] = {
            'C' : 0.1,
            'gamma' : 0.1
          }
          TrainMyClassifierParameters[1] = 'SVM'
      2. For RVM:
        TrainMyClassifierParameters[0] = {
            'alpha' : 1e-06,
            'beta' : 1e-06
        }
          TrainMyClassifierParameters[1] = 'RVM'     
      3. For GP:
          TrainMyClassifierParameters[0] = {
            'length_scale' : 10
          }
          TrainMyClassifierParameters[1] = 'GP'      
  Return Values:
    y_pred: Class labels on the validation set
    scores: Mean accuracy on the given test data and labels for this particular test prediction
    params: Trained Model for the required algorithm. Contains the estimated parameters and hyperparameters within the model. 
    
2. def MyCrossValidate(XTrain,YTrain2,Nf,Algorithm)
Input Parameters:
    XTrain: Input of the training data
    YTrain2: Labels of the training data XTrain
    Nf: Number of folds for cross-validation.(Nf=5)
    Algorithm: 'SVM'/'RVM'/'GP'
    
Return Values:
    YTrain: The	class labels for each validation sample
    EstParameters: Aarray of Estimated Parameters (Trained Model) for each Vn
    EstConfMatrices: Array of Confusion	Matrices for each Vn
    ConfMatrix: The overall Confusion Matrix
    
3. def MyConfusionMatrix(Y,YValidate,ClassNames)
Input Parameters:
    Y: Predicted by our model
    YValidate: Actual labels on validation set
    ClassNames: Array of names of every class. Example in our case: ['One','Two','Three','Four','Five']
Return Values:
    c_m: Classfication prediction in function of groundtruth as a fraction of actual class
    df: Same as c_m but formated in report ready form
    a_s: Returns accuracy classification score for groundtruth labels vs predicted labels for the overall 
    
4. def TestMyClassifier(XTest, Parameters, EstParameters)
Input Parameters:
    XTest: Test input data
    Parameters: Array of estimated parameters from the trained model
    EstParameters: Array of estimated hyperparameters from the trained model
Return Values:
    Ytest: Class labels of test data as predicted by the model.
