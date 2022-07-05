# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

1. Classification of the salary by analyzing the different attributes of the census bureau data such as race, sex , education ..etc
2. Logisic Regression model is used for prediction of the salary buckets.
3. Developed and maintained by Vivek S

## Intended Use

1. Intend to use if for research work to understand the impact of different peoples attributes on the economic growth
2. Not intend to establish link between race and salary which cause bias and discriminations
3. Intend to understand the causual impacts of the attributes

## Training Data

Census income data is multivariate data with mix of categorical and number data type from 1996.
This data is present in the UCI repository and donated by Ronny Kohavi and Barry Becker from UCI.
It consist of 14 attributes and target variable to predict whether income is more than 50 K or not.
Training data consist of 80% of the source dataset.
Note: data is not seprated by stratified sampling technique , hence chance of sample biasing.

## Evaluation Data

Evaluation data is 20% of the source dataset and generate by using train and test split package of the scikitlearn package.

## Metrics

Precision and recall are used to determine the effectiveness of the model to correctly classify the neagtive and positive classes. Confusion matrix is used to identify the true positive and true negative values of prediction of classes. Observed values for the evaluation metrics are : 

Precision:0.71
Recall:0.26 
fbeta:0.38



## Ethical Considerations

No special treatment and bias inducted in the training process of the model on the basis of individual specific attributes such as race, sex , marital status ..etc

__Data__: Model does not uses any sensitive data however there is bias in the data on the basis of race and gender
__Risks and harms__: There is a risk of relating the race and gender with the salary which can cause bias results towards prediction of salaries such as black females are high likely to be predicted under 50K salary than white males.

## Caveats and Recommendations

Further enhancement in validation technique can be done to achieve better results and concrete model evaluation results.
