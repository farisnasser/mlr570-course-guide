:::plain
**The flow:** training / validation / testing → underfitting and overfitting → K-fold cross-validation → confusion matrix → accuracy, precision, recall, F1 → decision threshold → weighted F-beta → regression metrics (MSE and MAE) → quick reference.
:::

Model evaluation asks whether a trained model has learned a pattern that **generalizes** beyond the data it saw during training. Training performance alone is not enough; we need held-out data and metrics that match the task.

:::ml Why this matters in ML
A model can achieve extremely low training loss and still perform badly on new data. Evaluation is therefore not just "calculate one number"; it is a **workflow** for separating learning, tuning, and final judgment.
:::

## 2.3.1 Training, Validation, and Testing

| | |
|---|---|
| **Training set** | Used to **fit model parameters** by minimizing training error. The model directly learns from these examples. |
| **Validation set** | Used during development for **model selection, hyperparameter tuning**, and monitoring generalization. |
| **Test set** | Used **only after** training / validation to estimate final performance on unseen data. |

:::ex Lecture example: 1000 samples
A possible split is 70% training (700 samples), 15% validation (150), and 15% testing (150). The exact ratio is task-dependent; the conceptual separation of roles is what matters.

:::fig split-bar
:::
:::

:::def Memory aid
**Train** = learn. $\quad$ **Validation** = tune / choose. $\quad$ **Test** = final judge.
:::

:::add Added analogy
Think of training as **practice questions**, validation as a **mock exam** used to adjust your strategy, and the test set as the **final exam**. Looking at the final exam while preparing would destroy its role as an unbiased final check.
:::

This is the same rule as fitting a scaler on training data only (§2.1.4): nothing about the test set may leak into decisions made during development.

## 2.3.2 Underfitting and Overfitting

:::def Underfitting
Model is **too simple** (or insufficiently trained) to capture the structure. Training error is **high** and validation / test error is also **high**.
:::

:::def Overfitting
Model fits the training data **too specifically**, including noise. Training error is **low**, but validation / test error is **much higher**.
:::

:::fig complexity
:::

:::cap
Lecture Figure 3.1 concept: training error tends to decrease as complexity increases, while validation error eventually rises again. The useful operating point is near the **minimum validation error**.
:::

:::fig three-fits
:::

:::cap
Lecture Figure 3.2 concept: too-simple fit, useful fit, and overly flexible fit. The same idea applies to regression curves and classification decision boundaries.
:::

:::add Added connection
**Bias–variance connection:** underfitting is associated with **high bias** (too rigid); overfitting is associated with **high variance** (too sensitive to the particular training sample). Same trade-off as the rolling-window size in §2.1.2.
:::

:::viz complexity | Raise the polynomial degree. Each fit is real least squares on the training points; the curves record training and validation error for every degree.
:::

:::ex Lecture example: choosing a model
| Model | Training error | Validation error | Interpretation |
|:-:|:-:|:-:|---|
| **1** | 12% | **11%** | best validation result |
| 2 | 17% | 18% | worse than Model 1 |
| 3 | **6%** | 19% | very low train error, but overfits |
| 4 | 23% | 23.5% | underfits / weak overall |
| 5 | 11% | 17% | some overfitting |

The lecture example deliberately shows why **minimum training error is not the selection criterion**. Model 3 has the lowest training error, but Model 1 has the lowest validation error and is the better choice.

Quick read: a big gap between train and validation $\to$ overfitting (Models 3, 5). Both high $\to$ underfitting (Model 4).
:::

## 2.3.3 K-Fold Cross-Validation

:::def K-fold cross-validation
Splits the data into $K$ subsets (folds), trains the model $K$ times, and **rotates** which fold is held out for validation. Final performance is the **average** across folds.
:::

:::fig kfold
:::

:::cap
Simplified reconstruction of the lecture $K=5$ diagram: each fold is the validation fold once, while the other four folds are used for training.
:::

Average validation error:

$$\boxed{\text{CV error}=\frac1K\sum_{i=1}^K\text{Error}_i}$$

:::ml Why use it?
It reduces dependence on one lucky or unlucky split, uses limited data efficiently (every sample is used for validation exactly once), and gives a more stable estimate of validation performance.
:::

:::add Added clarification
Larger $K$ means more training runs and therefore more computation. Each held-out fold is also smaller. The point is not that more folds are always "better"; it is another **reliability-versus-cost trade-off**.
:::

:::viz kfold | Choose $K$ and step through the runs. Each run fits on the training folds and measures the error on the held-out fold; the CV error is their average.
:::

For classification with imbalanced classes, folds are usually made **stratified** so each fold keeps the same class proportions (§2.1.3).

## 2.3.4 Classification Metrics

All of the basic binary classification metrics start from the **confusion matrix**.

:::fig confusion
:::

:::cap
Lecture Figure 3.4 concept. TP/TN lie on the correct diagonal; FP and FN are the two different kinds of error.
:::

|{stack}
| **Metric** | Formula | Question it answers |
|---|---|---|
| **Accuracy** | $\dfrac{TP+TN}{TP+TN+FP+FN}$ | Out of everyone, how many predictions were correct? |
| **Precision** | $\dfrac{TP}{TP+FP}$ | When I predict positive, how often am I right? |
| **Recall** | $\dfrac{TP}{TP+FN}$ | Of all real positives, how many did I find? |
| **F1** | $\dfrac{2PR}{P+R}$ | Are precision and recall both reasonably good? |

:::def Memory
**Precision:** denominator = everything **predicted** positive $(TP+FP)\to$ a column of the matrix.\
**Recall:** denominator = everything **actually** positive $(TP+FN)\to$ a row of the matrix.
:::

:::add Added clarification
**Accuracy warning:** on an imbalanced dataset, predicting only the majority class can produce high accuracy while completely failing on the minority class.

E.g. 95 negatives, 5 positives, model always says "negative": accuracy $=95/100=95\%$, but recall $=0/5=0$.
:::

Precision and recall are conditional probabilities (§1.2.4): precision $=P(\text{actual}+\mid\text{predicted}+)$, recall $=P(\text{predicted}+\mid\text{actual}+)$. Same two events, conditioned in opposite directions.

## 2.3.5 F1 and the Harmonic Mean

:::def F1
F1 is the **harmonic mean** of precision and recall, so a high value requires **both** metrics to be reasonably high.

$$F_1=\frac{2\cdot\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$$
:::

### Why harmonic mean?

If Precision $=0.1$ and Recall $=0.9$, the arithmetic mean is 0.5, which looks deceptively moderate. The harmonic mean gives

$$F_1=\frac{2(0.1)(0.9)}{0.1+0.9}=0.18,$$

strongly exposing the weak precision. The lecture also notes

$$\text{Harmonic Mean}\le\text{Geometric Mean}\le\text{Arithmetic Mean},$$

with equality only when the two inputs are equal. (Here: $0.18\le\sqrt{0.09}=0.3\le0.5$.)

:::ex Lecture examples
| Case | TP | FN | FP | TN | Precision | Recall | F1 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Balanced | 50 | 10 | 10 | 50 | 0.833 | 0.833 | 0.833 |
| High P, low R | 10 | 40 | 2 | 48 | 0.833 | 0.200 | 0.320 |
| High R, low P | 45 | 5 | 40 | 10 | 0.529 | 0.900 | 0.666 |

These lecture examples show why looking at precision or recall alone can be misleading. F1 drops when one side is much weaker than the other.
:::

:::viz means | Slide precision and recall, or load a lecture row. The harmonic mean (F1) always sits at or below the geometric and arithmetic means.
:::

## 2.3.6 The Decision Threshold

:::add Added note
The course outline includes the decision threshold, but the revised lecture PDF does not contain a dedicated threshold subsection. This explanation is therefore an added clarification.
:::

A probabilistic binary classifier often outputs a score such as $P(y=1\mid\mathbf{x})=0.73$. A threshold $t$ converts that score into a class:

$$\boxed{\text{score}\ge t\;\to\;\text{predict positive}\qquad\qquad\text{score}<t\;\to\;\text{predict negative}}$$

:::fig threshold
:::

|{stack}
| **Change** | What happens | Typical effect |
|---|---|---|
| **Lower threshold** | more samples called positive | Recall tends to rise; Precision tends to fall |
| **Higher threshold** | fewer samples called positive | Precision tends to rise; Recall tends to fall |

:::def Key idea
Changing the threshold changes TP, FP, TN, and FN, so it changes accuracy, precision, recall, and F1 **without retraining the model**.
:::

:::viz threshold | Slide the threshold $t$ through the scores of 100 actual-positive and 100 actual-negative samples. The confusion matrix and every metric are counted from the samples; $\beta$ sets the F-beta weight (§2.3.7).
:::

## 2.3.7 Weighted F-beta

:::def F-beta
Generalizes F1 so that precision and recall do not have to receive equal emphasis.

$$F_\beta=\frac{(1+\beta^2)(\text{Precision}\cdot\text{Recall})}{\beta^2\,\text{Precision}+\text{Recall}}$$
:::

- $\beta=1$ gives F1.
- $\beta>1$ emphasizes **recall** (e.g. disease screening: missing a positive is worse).
- $\beta<1$ emphasizes **precision** (e.g. spam filter: a false alarm is worse).

:::ex Lecture example
$TP=45$, $FN=40$, $FP=5$, $TN=10$ gives

$$\text{Precision}=\frac{45}{50}=0.9,\qquad\text{Recall}=\frac{45}{85}=0.529,\qquad F_1=0.666.$$

With $\beta=2$:

$$F_2=\frac{5(0.9)(0.529)}{4(0.9)+0.529}=\frac{2.38}{4.13}=0.577.$$

Because recall is the weaker metric, emphasizing recall pulls the score downward.
:::

## 2.3.8 Regression Metrics: MSE and MAE

For regression we compare a continuous prediction $\hat y_i$ with its true value $y_i$. The lecture focuses on MSE and MAE.

### Mean Squared Error (MSE)

:::def MSE
$$\text{MSE}=\frac1N\sum_{i=1}^N(y_i-\hat y_i)^2$$

MSE **squares** errors. Large mistakes therefore receive disproportionately large penalties. It is smooth and convenient for gradient-based optimization, but highly sensitive to outliers.
:::

### Mean Absolute Error (MAE)

:::def MAE
$$\text{MAE}=\frac1N\sum_{i=1}^N|y_i-\hat y_i|$$

MAE grows **linearly** with error. It is more robust to outliers and remains in the **original units** of the target, making it easier to interpret.
:::

| **Metric** | Penalty | Outliers | Units |
|---|---|---|---|
| **MSE** | squared | very sensitive | squared target units |
| **MAE** | absolute / linear | more robust | same as target |

This is the $L_2$ vs $L_1$ story again (§1.1.7, §2.2.2): squaring lets one big error dominate.

:::ex Lecture example 1
$y=[3,\ 0.5,\ 2,\ 7]$, $\ \hat y=[2.5,\ 0,\ 2,\ 8]$. Errors: $[0.5,\ 0.5,\ 0,\ -1]$.

$$\text{MSE}=\frac{0.25+0.25+0+1}{4}=0.375\qquad\qquad\text{MAE}=\frac{0.5+0.5+0+1}{4}=0.5$$

They should not be compared directly because MSE is in squared units. RMSE $=\sqrt{\text{MSE}}\approx0.61$ is in the original units.
:::

:::ex Lecture example 2: add one outlier
$y=[3,\ 0.5,\ 2,\ 7,\ 100]$, $\ \hat y=[2.5,\ 0,\ 2,\ 8,\ 8]$. The final error is 92.

$$\text{MSE}=\frac{1.5+92^2}{5}=\frac{1.5+8464}{5}=1693.1\qquad\qquad\text{MAE}=\frac{2+92}{5}=18.8$$

Squaring 92 produces 8464, which explains why MSE reacts so dramatically.
:::

:::ml Which one to use
**Use MSE** when large mistakes deserve especially strong punishment or smooth gradient-based optimization is important.\
**Use MAE** when robustness to extreme errors and interpretability in the original units matter more.
:::

Same idea as mean vs median (§2.1.2): if you predict one constant for all points, MSE is minimized by the **mean** and MAE by the **median**, which is why MAE shrugs off outliers.

:::viz mse-mae | Drag the last target $y_5$ (or load the lecture examples). On the right, one constant prediction $c$ for all points: MSE bottoms out at the mean, MAE at the median.
:::

## 2.3.9 Quick Reference

|{stack}
| **Concept** | Formula / rule | Meaning |
|---|---|---|
| **Training** | fit model parameters | data the model learns from |
| **Validation** | tune / select | development-time generalization check |
| **Test** | final evaluation | unseen final benchmark |
| **Underfit** | train error high; val error high | model too simple / insufficiently trained |
| **Overfit** | train error low; val error high | fits training data too specifically |
| **K-fold CV** | rotate held-out fold; average | more stable validation estimate |
| **Accuracy** | $(TP+TN)/\text{all}$ | overall fraction correct |
| **Precision** | $TP/(TP+FP)$ | correctness of positive predictions |
| **Recall** | $TP/(TP+FN)$ | coverage of actual positives |
| **F1** | $2PR/(P+R)$ | harmonic balance of P and R |
| **Threshold** | score $\ge t\to+$ | trades precision vs recall, no retraining |
| **F-beta** | $\beta>1$ recall; $\beta<1$ precision | weighted P/R balance |
| **MSE** | $\frac1N\sum(y_i-\hat y_i)^2$ | strongly punishes large errors |
| **MAE** | $\frac1N\sum\lvert y_i-\hat y_i\rvert$ | robust, original units |

> **Source note.** The core content, examples, figures, and ordering in these notes follow the MLR570 Chapter 2 revised lecture notes. Amber boxes mark study clarifications or analogies that were added during our walkthrough so they remain distinguishable from source-derived material.
