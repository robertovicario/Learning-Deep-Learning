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
        <td bgcolor="green">True Positives (TP)</td>
        <td bgcolor="red">False Negatives (FN)</td>
    </tr>
    <tr>
        <td><strong>Actual</strong></td>
        <td bgcolor="red">False Positives (FP)</td>
        <td bgcolor="green">True Negatives (TN)</td>
    </tr>
</table>

### Metrics

Several metrics can be derived from the confusion matrix to evaluate the model's performance.

**Accuracy:** Measures the proportion of correctly predicted instances out of the total instances:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision:** Indicates the proportion of positive identifications that were actually correct:

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall:** Measures the proportion of actual positives that were correctly identified:

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1 Score:** The harmonic mean of precision and recall, providing a single metric that balances both:

$$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### ROC

Receiver Operating Characteristic (ROC) curves plot the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. The ROC curve illustrates the trade-off between sensitivity and specificity.

**True Positive Rate (TPR):** Also known as recall:

$$\text{TPR} = \frac{TP}{TP + FN}$$

**False Positive Rate (FPR):** Measures the proportion of actual negatives that were incorrectly classified as positive:

$$\text{FPR} = \frac{FP}{FP + TN}$$

### AUC

The Area Under the ROC Curve (AUC) provides a single value to summarize the performance of the model. AUC ranges from 0 to 1, with higher values indicating better performance.

<table>
<tr>
    <td><img src="/metrics/img/1.png" width="512"></td>
</tr>
<tr>
    <td align="center">AUC</td>
</tr>
</table>

## Reference

- [Stanford CS230 Machine Learning Cheatsheet](https://stanford.edu/~shervine/teaching/cs-229) by Afshine Amidi and Shervine Amidi