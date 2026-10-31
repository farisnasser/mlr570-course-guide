:::plain
**The flow:** Why preprocess at all → aggregation → sampling → normalization → encoding → discretization → dimensionality reduction → PCA → feature selection.
:::

Raw data is rarely ready to give directly to a machine-learning model. Measurements may be noisy, incomplete, differently scaled, or represented in incompatible formats. **Preprocessing** turns that raw data into a cleaner, more consistent representation that the model can actually use.

:::ml Why preprocessing matters in ML
Good preprocessing helps a model learn meaningful patterns, generalize to unseen data, converge more efficiently during training, and avoid problems such as one feature dominating simply because its numerical scale is much larger than another.
:::

## 2.1.1 Why Preprocess at All?

The basic pipeline is

:::fig pipeline
:::

The major operations in this part are:

- **Aggregation:** combine several observations into a summary.
- **Sampling:** select a representative subset of a larger dataset.
- **Normalization:** rescale numerical features to comparable ranges.
- **Encoding:** convert categorical values into numerical representations.
- **Discretization:** convert continuous values into a finite set of bins.
- **Dimensionality reduction / feature selection:** reduce the number of dimensions while keeping useful information.

:::ex Scale can distort geometry
Suppose Feature 1 has values around $10^4$ while Feature 2 has values around 10. A distance-based method can become dominated by Feature 1, not because Feature 1 is more informative, but simply because its numbers are much larger. Scaling can restore a geometry in which both features are comparable.\
E.g. for two points differing by 1000 in Feature 1 and by 5 in Feature 2:

$$\|\mathbf{x}-\mathbf{y}\|_2=\sqrt{1000^2+5^2}\approx1000.01$$

Feature 2 contributes almost nothing to the distance (§1.1.7).
:::

:::viz scale | Drag the query point. The ring marks its nearest neighbour under the chosen geometry; switch between the raw features and the two scalings.
:::

## 2.1.2 Aggregation

:::def Aggregation
Summarizes or combines data points in order to reduce size or complexity and expose higher-level structure such as trends.
:::

Common summaries are

$$\text{sum}=\sum_{i=1}^nx_i,\qquad\text{mean}=\frac1n\sum_{i=1}^nx_i,\qquad\text{median}=\text{middle ordered value}.$$

The **mean** is simple and efficient but sensitive to outliers. The **median** is more robust to extreme values, although it throws away more fine-grained variation. Aggregation can also be spatial or windowed, e.g. hourly temperatures $\to$ daily average temperatures.

:::ex Mean vs median
For $\{2,3,3,4,100\}$: $\ \text{mean}=112/5=22.4,\ \ \text{median}=3$.\
One outlier drags the mean far away from where most of the data sits; the median doesn't move.
:::

### Rolling / windowed aggregation

For a univariate time series $\{x_t\}_{t=1}^T$ and a right-aligned window of size $w$,

$$\boxed{\bar x_t=\frac1w\sum_{i=0}^{w-1}x_{t-i}}\qquad\qquad\boxed{\tilde x_t=\operatorname{median}(x_{t-w+1},\dots,x_t)}$$

The window uses the current and previous $w-1$ samples; **no future sample is used**. Therefore the first $w-1$ outputs are undefined until a complete window exists.

:::fig rolling
:::

:::ml What rolling aggregation does
Averaging suppresses high-frequency noise and makes the slower trend easier to see. A rolling median does the same while being less affected by isolated spikes.
:::

:::add Added intuition — the bias–variance trade-off here
A **small** window reacts quickly to changes, so it has less lag (lower bias) but follows noise more strongly (higher variance). A **large** window is smoother (lower variance) but contains more old values, so it lags behind a changing signal (higher bias). Think: *responsive but noisy $\leftrightarrow$ smooth but laggy.*
:::

Larger $w$ therefore gives stronger smoothing and trend extraction, but it can hide short-term dynamics. There is no universally optimal window size; it depends on the signal-to-noise ratio, the domain, and the downstream task.

:::viz aggregate | Drag the outlier in $\{2,3,3,4,100\}$; then set the window $w$ on the series and compare the rolling mean with the rolling median around the spikes.
:::

## 2.1.3 Sampling

:::def Sampling
Selects a subset of a larger dataset or population. The goal is usually to reduce computational cost while keeping the subset **representative**; sampling can also be used to address class imbalance.
:::

|{stack}
| **Method** | Main idea | Main risk / limitation |
|---|---|---|
| **Random** | Every observation has equal probability of selection. | Rare subgroups may be missed. |
| **Stratified** | Split into known strata and sample proportionally from each. | Requires strata / class labels. |
| **Oversampling** | Increase the minority class by duplication or synthetic generation. | Overfitting or unrealistic synthetic samples. |
| **Undersampling** | Remove observations from the majority class. | Useful information can be discarded. |
| **Systematic** | Select every $k$th observation after a random start. | Hidden periodicity can introduce bias. |

:::ex Stratified sampling
If only 5% of a medical dataset belongs to a rare positive class, a stratified train/validation split keeps roughly 5% positives in each subset instead of leaving that proportion to chance.
:::

:::viz stratified | Split the same dataset (5% positives) many times. Each dot is one validation set's share of positives: random splits scatter, stratified splits sit at 5%.
:::

:::ml Imbalanced classification
Oversampling makes the minority class larger; undersampling makes the majority class smaller. Both can produce a more balanced training set, but they change the data in different ways. In sensitive applications, synthetic minority samples should only be used when their realism is defensible.
:::

:::fig resampling
:::

### Advanced strategies — recognize the names

- **Importance sampling:** chooses observations according to task-specific importance.
- **Cluster sampling:** selects groups rather than individuals.
- **Reservoir sampling:** maintains a random sample from a stream of unknown length.
- **Adaptive sampling:** focuses on informative regions.
- **Synthetic methods** such as SMOTE or GAN-based methods: generate new samples.

## 2.1.4 Normalization

:::def Normalization / scaling
Puts numerical features on comparable scales so that a large numerical magnitude is **not mistaken for greater importance**.
:::

This matters particularly for distance-based methods such as $k$-NN and K-means, and for gradient-based optimization where poor scaling can slow or destabilize convergence (§1.3.6).

### Min–Max scaling

Usually maps a feature to $[0,1]$:

$$\boxed{x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}}$$

The minimum becomes 0, the maximum becomes 1, and values in between are mapped proportionally. It is simple and intuitive, but because $x_{\min}$ and $x_{\max}$ define the entire scale, **extreme outliers can distort the transformation**.

:::ex Example
Ages $\{20,30,40,60\}$: $\ x_{\min}=20$, $x_{\max}=60$, so $30\mapsto\dfrac{30-20}{60-20}=0.25$.\
Add one outlier age of 100 and now $30\mapsto\dfrac{10}{80}=0.125$: every normal value gets squeezed toward 0.
:::

### Z-score standardization

Centers and rescales using the feature mean and standard deviation (§1.2.2):

$$\boxed{x'=\frac{x-\mu}{\sigma}}$$

After standardization the feature has **mean 0 and standard deviation 1** in the data used to fit the scaler. A standardized value of $x'=2$ means the observation lies two standard deviations above the fitted mean.

:::viz scalers | Drag any age (or add the outlier 100). The two lines below are the same ages after min–max scaling and after z-scores.
:::

:::ex Why scaling matters
Age may be measured on the order of $10^1$, income on the order of $10^5$, and credit score below $10^3$. Without scaling, the largest-magnitude feature can dominate geometric calculations even if it is not the most predictive feature.
:::

:::def Practical rule: fit on training data only
Compute $x_{\min},x_{\max}$ or $\mu,\sigma$ from the **training set**, then use those same fitted values to transform training, validation, and test data.
:::

For example,

$$x'_{\text{test}}=\frac{x_{\text{test}}-\mu_{\text{train}}}{\sigma_{\text{train}}},$$

not a new test-set mean and standard deviation. The validation/test sets are meant to behave like **unseen data**; computing statistics on them leaks information about the test set into training.

Further methods mentioned in the lecture include robust scaling (median/IQR), log or power transforms, L2 unit-vector normalization, batch/layer normalization for neural-network activations, and whitening.

## 2.1.5 Encoding

:::def Encoding
Converts categorical values into numerical representations that a machine-learning model can process.
:::

### Label encoding

Assign each category an integer, e.g.

$$\text{Red}\to0,\qquad\text{Green}\to1,\qquad\text{Blue}\to2.$$

The danger is that this may create an **artificial ordinal relationship**: the model can interpret $2>1>0$ as meaningful even when the categories have no natural order.

### One-hot encoding

Create one binary column for each category:

$$\text{Red}\to[1,0,0],\qquad\text{Green}\to[0,1,0],\qquad\text{Blue}\to[0,0,1].$$

This works well for low-cardinality nominal categories. The main limitation is **dimensionality**: thousands of categories create thousands of sparse columns.

:::add Added intuition — why one-hot is preferred in linear-algebra terms
For $k$ unordered categories, one-hot encoding represents the categories as the **standard basis vectors** of $\mathbb{R}^k$:

$$\text{Red}=e_1=\begin{bmatrix}1\\0\\0\end{bmatrix},\qquad\text{Green}=e_2=\begin{bmatrix}0\\1\\0\end{bmatrix},\qquad\text{Blue}=e_3=\begin{bmatrix}0\\0\\1\end{bmatrix}.$$

Different basis vectors are orthogonal,

$$e_i^Te_j=0\qquad(i\neq j),$$

and every pair is equally separated,

$$\|e_i-e_j\|_2=\sqrt2\qquad(i\neq j).$$

So the representation introduces **no fake ordering or magnitude**. It also gives a linear model one independent coefficient per category:

$$\boxed{\mathbf{w}^Te_j=w_j}$$

The active one-hot coordinate simply selects the parameter belonging to that category.
:::

Compare label encoding: $|0-2|=2$ but $|0-1|=1$, so Red looks "twice as far" from Blue as from Green, which means nothing.

:::viz onehot | Drag to orbit. Label encoding puts the colours on one line with unequal gaps; one-hot puts them on the three axes, every pair $\sqrt2$ apart.
:::

### Frequency encoding

Replace a category by how often it appears in the dataset. This can reduce dimensionality for high-cardinality features, but frequency itself does not necessarily imply a useful relationship with the target.

### Binary encoding

Assign categories integers, convert those integers to binary, then expand the binary digits into columns. It can use fewer columns than one-hot encoding when the number of categories is large, but it is less interpretable.

:::ex Column count
For $k$ categories: one-hot needs $k$ columns, binary needs about $\lceil\log_2k\rceil$.\
E.g. $k=1000$: one-hot $\to$ 1000 columns, binary $\to$ 10 columns ($2^{10}=1024$).
:::

## 2.1.6 Discretization

:::def Discretization
Converts a continuous variable into a finite set of intervals or bins. It trades numerical precision for a simpler, often more robust or interpretable representation.
:::

### Equal-width binning

Divide the numerical range into $k$ intervals of identical width:

$$\boxed{\text{width}=\frac{x_{\max}-x_{\min}}{k}}$$

Every interval spans the same numerical distance, but the **number of observations in each bin can be very different**.

### Equal-frequency (quantile) binning

Sort the values and choose boundaries so that each bin contains approximately the same number of observations. Thus

$$\boxed{\text{similar counts per bin, different numerical widths.}}$$

:::fig binning
:::

### K-means binning

Use K-means to partition the values into $k$ clusters by minimizing within-cluster variation. The bin boundaries therefore **adapt to the empirical data distribution**.

|{stack}
| **Method** | What is controlled? | Main trade-off |
|---|---|---|
| **Equal-width** | Same interval width | Simple, but counts can be very uneven. |
| **Equal-frequency** | Roughly same count per bin | Balanced counts, but widths vary. |
| **K-means** | Data-driven cluster compactness | Adaptive, but less transparent. |

:::viz binning | Choose $k$ and drag the points; the three rows bin the same values by equal width, equal frequency and K-means.
:::

## 2.1.7 Dimensionality Reduction

:::def Dimensionality reduction
Maps high-dimensional data into a lower-dimensional space while trying to preserve useful structure.
:::

Main motivations are lower computational/storage cost, noise reduction, visualization in 2D/3D, and improved generalization by removing redundant variation.

The lecture mentions PCA, LDA, t-SNE, UMAP, and random projection, but develops **PCA** as the principal technique in this section.

## 2.1.8 Principal Component Analysis (PCA)

:::def PCA
Finds a new set of **orthogonal axes**, called **principal components**, such that the projected data retains as much variance as possible.
:::

Given

$$X\in\mathbb{R}^{n\times d}$$

with $n$ samples and $d$ features, PCA finds $k<d$ orthogonal directions

$$W_k\in\mathbb{R}^{d\times k}$$

and produces

$$\boxed{Y=X_cW_k\in\mathbb{R}^{n\times k}}$$

:::fig pca-cloud
:::

:::ml Geometric picture
PCA **rotates the coordinate system** so that PC1 points along the direction where the data cloud varies most. PC2 is orthogonal to PC1 and captures the next-largest remaining variance, and so on.
:::

### Step 1 — center the data

Subtract the mean of each feature:

$$\boxed{X_c=X-\boldsymbol\mu}$$

PCA cares about variation around the mean, so centering moves the data cloud to the origin without changing its shape.

### Step 2 — build the covariance matrix

$$\boxed{\Sigma=\frac{1}{n-1}X_c^TX_c\in\mathbb{R}^{d\times d}}$$

For two features,

$$\Sigma=\begin{bmatrix}\operatorname{Var}(x_1)&\operatorname{Cov}(x_1,x_2)\\\operatorname{Cov}(x_2,x_1)&\operatorname{Var}(x_2)\end{bmatrix}.$$

The **diagonal** tells us how much each feature varies; the **off-diagonal** terms tell us how features move together (§1.2.7). Thus $\Sigma$ summarizes the orientation and spread of the data cloud. It is always symmetric, since $\operatorname{Cov}(x_1,x_2)=\operatorname{Cov}(x_2,x_1)$.

:::add Added intuition — why $X_c^TX_c$?
Multiplying $X_c^T$ by $X_c$ compares feature columns with one another. Diagonal terms contain sums of squares such as $\sum_ix_{i1}^2$; off-diagonal terms contain cross-products such as $\sum_ix_{i1}x_{i2}$. Because the features are centered, dividing these quantities by $n-1$ gives variances and covariances.
:::

### Step 3 — eigenvalue decomposition

Solve

$$\boxed{\Sigma\mathbf{v}_i=\lambda_i\mathbf{v}_i}$$

Each eigenvector $\mathbf{v}_i$ gives a principal-component **direction**; its eigenvalue $\lambda_i$ gives the **variance along that direction** (§1.1.5). Therefore

$$\boxed{\text{eigenvector}=\text{direction},\qquad\text{eigenvalue}=\text{variance captured.}}$$

Sort the eigenvalues so that $\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_d$.

### Step 4 — keep the top $k$ directions and project

Form

$$W_k=[\,\mathbf{v}_1\ \mathbf{v}_2\ \cdots\ \mathbf{v}_k\,]$$

and compute

$$\boxed{Y=X_cW_k}$$

With samples stored as rows, the $i$th sample transforms as

$$\boxed{\mathbf{y}_i=\mathbf{x}_iW_k}$$

Each entry of $\mathbf{y}_i$ is the component of the original centered sample along one PCA axis.

:::add Added intuition — resolving a force vector
In mechanics, a force is resolved into components along chosen axes, e.g. horizontal and vertical. PCA does the same geometric operation: it resolves each data vector into components along PC1, PC2, …. The difference is that PCA **learns the axes from the data** by choosing directions of maximum variance.
:::

:::viz pca | Turn the candidate axis: every point drops a perpendicular onto it and the spread of the landing spots is the projected variance. It peaks on the top eigenvector of $\Sigma$ (PC1). Drag points to reshape the cloud.
:::

### Why this becomes dimensionality reduction

If all $d$ components are kept, PCA is mainly a **change of coordinate system**. If only the strongest $k<d$ components are retained, low-variance directions are discarded:

$$\boxed{d\text{ original features}\longrightarrow k\text{ PCA coordinates.}}$$

This is closely related to **diagonalization** (§1.1.6): PCA uses the eigenvectors of the covariance matrix as the natural coordinate axes. In that basis,

$$W^T\Sigma W=\Lambda,$$

where $\Lambda$ is diagonal. A diagonal covariance means the new PCA features are uncorrelated with each other.

### Explained variance

The fraction of total variance explained by component $i$ is

$$\boxed{\text{EVR}_i=\frac{\lambda_i}{\sum_j\lambda_j}}$$

:::ex Example
:::cols
:::col
If eigenvalues are $70,20,10$, then PC1 explains 70%, PC2 explains 20%, and PC3 explains 10%.\
Keeping PC1 and PC2 preserves **90%** of the total variance.
:::
:::col
:::fig evr-bars
:::
:::
:::
:::

### Choosing $k$

The lecture gives three common approaches:

- **Elbow method:** choose the point after which extra components add little variance.
- **Variance threshold:** keep enough components to capture a target such as 95% cumulative variance.
- **Cross-validation:** choose the $k$ that performs best on the downstream task.

### PCA via SVD

Instead of explicitly forming the covariance matrix, PCA can be computed from

$$\boxed{X_c=USV^T}$$

where the columns of $V$ are the principal-component directions. This directly connects PCA to the SVD studied in Chapter 1 (§1.1.8).

:::add Added note
Why it works:

$$X_c^TX_c=(USV^T)^T(USV^T)=VS^TU^TUSV^T=VS^2V^T\qquad(\text{since }U^TU=I)$$

$$\Rightarrow\quad\Sigma=V\,\frac{S^2}{n-1}\,V^T$$

which is exactly an eigendecomposition of $\Sigma$: the columns of $V$ are its eigenvectors, and $\lambda_i=\sigma_i^2/(n-1)$.
:::

:::ml Scale before PCA
PCA is variance-based, so a feature measured on a much larger numerical scale can dominate the components. It is therefore standard to **standardize features before PCA** when their scales differ substantially (§2.1.4).
:::

### Limitations

PCA captures **linear** structure, can be sensitive to outliers, depends on scaling, and often reduces interpretability because each principal component is a linear combination of the original features.

## 2.1.9 Feature Selection

:::def Feature selection
Keeps the most useful subset of the **original** features. Unlike PCA, it does not create new synthetic features.
:::

For example,

$$[x_1,x_2,x_3,x_4,x_5]\longrightarrow[x_1,x_3,x_5].$$

The goals are improved generalization, lower training/inference cost, and better interpretability.

|{stack}
| | PCA | Feature selection |
|---|---|---|
| **Output** | new features (linear combinations) | a subset of the original features |
| **Interpretability** | lower | kept |
| **Uses the target?** | no | filter: sometimes; wrapper: yes |

### Filter methods

Filter methods score features using statistical criteria **independently of a predictive model**.

**Variance threshold.**

$$\operatorname{Var}(X_j)=\frac1n\sum_{i=1}^n(X_{ij}-\mu_j)^2.$$

Very low variance suggests the feature changes little across samples. Limitation: variance alone ignores the target.

**Correlation with the target.**

$$\rho(X_j,y)=\frac{\operatorname{Cov}(X_j,y)}{\sigma_{X_j}\sigma_y}.$$

Large $|\rho|$ indicates strong **linear** association, but nonlinear dependence can be missed (§1.2.7).

**Mutual information.**

$$I(X_j;y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)\,p(y)}.$$

High mutual information means knowing $X_j$ strongly reduces uncertainty about the target; it can capture **nonlinear as well as linear** dependence.

If $X_j$ and $y$ are independent, then $p(x,y)=p(x)p(y)$ in every cell (§1.2.5), so every log term is $\log1=0$ and $I=0$. Mutual information measures how far the joint is from that independent case.

:::ml Filter-method trade-off
They are fast, scalable, interpretable, and model-independent, but they can miss interactions between features and may retain redundant features.
:::

### Wrapper methods

Wrapper methods evaluate feature subsets by **repeatedly training and validating a predictive model**. They can capture feature interactions, but they are computationally expensive.

- **Recursive Feature Elimination (RFE):** train on all features $\to$ rank importance $\to$ remove the least important $\to$ repeat.
- **Forward selection:** start with no features and repeatedly add the feature that improves validation performance the most.
- **Backward elimination:** start with all features and repeatedly remove the least useful feature.

The lecture groups feature-selection methods into **filter, wrapper, and embedded** approaches, but develops filter and wrapper methods in detail here.

## 2.1.10 Quick Reference

|{stack}
| **Concept** | Formula / definition | Meaning |
|---|---|---|
| **Aggregation** | $\bar x=\frac1n\sum_ix_i$ | Combine observations into a simpler summary. |
| **Rolling mean** | $\bar x_t=\frac1w\sum_{i=0}^{w-1}x_{t-i}$ | Smooth a time series using recent history. |
| **Sampling** | Select a subset of observations | Reduce cost or preserve/adjust population structure. |
| **Min–Max** | $x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}$ | Map to a fixed range, usually $[0,1]$. |
| **Z-score** | $x'=\frac{x-\mu}{\sigma}$ | Center at 0 and scale by standard deviation. |
| **Fit scaler** | Fit on train, reuse on val/test | Keep preprocessing based only on training information. |
| **One-hot** | category $j\to e_j$ | Separate unordered categories without fake magnitude/order. |
| **Discretization** | continuous $\to$ finite bins | Trade precision for simpler categories. |
| **Equal-width** | width $=\frac{x_{\max}-x_{\min}}{k}$ | Equal numerical intervals. |
| **Equal-frequency** | approx. equal counts per bin | Variable widths, balanced counts. |
| **PCA covariance** | $\Sigma=\frac{1}{n-1}X_c^TX_c$ | Summarize variance/covariance structure. |
| **PCA eigenproblem** | $\Sigma\mathbf{v}_i=\lambda_i\mathbf{v}_i$ | $\mathbf{v}_i$ = direction, $\lambda_i$ = captured variance. |
| **PCA projection** | $Y=X_cW_k$ | Resolve samples onto the top $k$ principal axes. |
| **Explained variance** | $\lambda_i/\sum_j\lambda_j$ | Fraction of total variance captured by PC $i$. |
| **PCA via SVD** | $X_c=USV^T$ | Columns of $V$ = principal directions. |
| **Feature selection** | keep original feature subset | Reduce dimensions without creating new features. |
| **Correlation filter** | $\rho(X_j,y)$ | Rank linear association with target. |
| **Mutual information** | $I(X_j;y)$ | Rank general statistical dependence with target. |
| **Wrapper method** | train/validate feature subsets | Captures interactions but costs more computation. |

:::def Big picture
Preprocessing is not just "cleaning." It changes the **representation** on which the learning algorithm operates. The core question throughout this part is: *what information should we preserve, suppress, rescale, or re-express before asking a model to learn?*
:::

> **Source note.** The core content and ordering in these notes follow the MLR570 Chapter 2 revised lecture notes. Amber boxes mark added intuitive explanations developed during study, so they can be distinguished from the lecture-source material.
