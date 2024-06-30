# Metrics

## Overview

Evaluating the performance of models is crucial to understanding their effectiveness and making informed decisions about their deployment. Various metrics are used to assess models based on their type and the specific requirements of the problem at hand.

## Classification Metrics

Classification metrics are used to evaluate the performance of classification algorithms, which predict categorical labels. 

### Confusion Matrix

A confusion matrix is a table used to evaluate the performance of a classification model. It summarizes the prediction results by comparing actual and predicted values. The table helps in understanding the types of errors the model is making.

<table>
    <tr>
        <td></td>
        <td><strong>Predicted</strong></td>
        <td><strong>Predicted</strong></td>
    </tr>
    <tr>
        <td><strong>Actual</strong></td>
        <td>True Positives<br>(TP)</td>
        <td>False Negatives<br>(FN)</td>
    </tr>
    <tr>
        <td><strong>Actual</strong></td>
        <td>False Positives<br>(FP)</td>
        <td>True Negatives<br>(TN)</td>
    </tr>
</table>

### Accuracy

Measures the proportion of correctly predicted instances out of the total instances:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Precision

Indicates the proportion of positive identifications that were actually correct:

$$\text{Precision} = \frac{TP}{TP + FP}$$

### Recall (Sensitivity)

Measures the proportion of actual positives that were correctly identified:

$$\text{Recall} = \frac{TP}{TP + FN}$$

### Specificity

Indicates the proportion of actual negatives that were correctly identified:

$$\text{Specificity} = \frac{TN}{TN + FP}$$

### F1 Score

The harmonic mean of precision and recall, providing a single metric that balances both:

$$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Receiver Operating Characteristic (ROC)

Receiver Operating Characteristic (ROC) curves plot the Sensitivity against the Specificity (Recall) at various threshold settings. The ROC curve illustrates the trade-off between sensitivity and specificity.

### Area Under the ROC Curve (AUC)

The Area Under the ROC Curve (AUC) provides a single value to summarize the performance of the model. AUC ranges from 0 to 1, with higher values indicating better performance.

<table>
<tr>
    <td><img src="/metrics/img/1.png" width="512"></td>
</tr>
<tr>
    <td align="center">AUC</td>
</tr>
</table>

## Regression Metrics

Regression metrics are used to evaluate the performance of regression algorithms, which predict continuous values. Understanding these metrics helps in determining how well the model captures the underlying patterns in the data.

### Total Sum of Squares (SST)

The Total Sum of Squares (SST) measures the total variance in the response variable $y$. It represents the total variability of the observed data points from their mean:

$$SS_{tot} = \sum_{i=1}^{m} (y_i - \bar{y})^2$$

Where:

- $y_i$ is the observed value.

- $\bar{y}$ is the mean of the observed values.

- $m$ is the number of observations.

### Explained Sum of Squares (SSR)

The Explained Sum of Squares (SSR) measures the amount of variance explained by the regression model. It represents the reduction in variability of $y$ due to the model:

$$SS_{reg} = \sum_{i=1}^{m} (f(x_i) - \bar{y})^2$$

Where:

- $f(x_i)$ is the predicted value.

### Residual Sum of Squares (SSE)

The Residual Sum of Squares (SSE) measures the variance that is not explained by the regression model. It represents the discrepancy between the observed data and the predicted values:

$$SS_{res} = \sum_{i=1}^{m} (y_i - f(x_i))^2$$

### Coefficient of Determination

The Coefficient of Determination, denoted as $R^2$, is a statistical measure that indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. It provides an indication of goodness of fit and is a value between 0 and 1:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

## Reference

- [Stanford CS230 Machine Learning Cheatsheet](https://stanford.edu/~shervine/teaching/cs-229) by Afshine Amidi and Shervine Amidi
