# Diagnostics

## Overview

Diagnostics play a crucial role in understanding and improving model performance. By diagnosing issues like bias and variance, practitioners can take appropriate actions to enhance their model's accuracy and generalization capabilities:

- **Bias:** Bias refers to the error introduced by approximating a real-world problem, which may be extremely complicated, by a much simpler model. High bias can cause the model to miss relevant relations between features and target outputs (underfitting).

- **Variance:** Variance refers to the error introduced by the model's sensitivity to small fluctuations in the training set. High variance can cause the model to model the random noise in the training data rather than the intended outputs (overfitting).

### Bias-Variance Tradeoff

The bias-variance tradeoff is a fundamental concept in machine learning that describes the tradeoff between a model's ability to minimize bias and variance to achieve good generalization performance. Understanding this tradeoff is essential for diagnosing model performance issues.

<table>
    <tr>
        <td></td>
        <td><strong>Underfitting</strong></td>
        <td><strong>Just Right</strong></td>
        <td><strong>Overfitting</strong></td>
    </tr>
    <tr>
        <td align="center"><strong>Symptoms</strong></td>
        <td>High bias, low variance. Poor performance on both training and test data. Model is too simple to capture underlying patterns.</td>
        <td>Balanced bias and variance. Good performance on both training and test data. Model complexity matches the complexity of the data.</td>
        <td>Low bias, high variance. Good performance on training data but poor performance on test data. Model is too complex and captures noise.</td>
    </tr>
        <tr>
        <td align="center"><strong>Regression</strong></td>
        <td align="center"><img src="/diagnostics/img/4.png" width="128"></td>
        <td align="center"><img src="/diagnostics/img/5.png" width="128"></td>
        <td align="center"><img src="/diagnostics/img/6.png" width="128"></td>
    </tr>
    <tr>
        <td align="center"><strong>Classification</strong></td>
        <td align="center"><img src="/diagnostics/img/1.png" width="128"></td>
        <td align="center"><img src="/diagnostics/img/2.png" width="128"></td>
        <td align="center"><img src="/diagnostics/img/3.png" width="128"></td>
    </tr>
    <tr>
        <td align="center"><strong>Deep Learning</strong></td>
        <td align="center"><img src="/diagnostics/img/7.png" width="128"></td>
        <td align="center"><img src="/diagnostics/img/8.png" width="128"></td>
        <td align="center"><img src="/diagnostics/img/9.png" width="128"></td>
    </tr>
</table>
