:::plain
**The flow:** linear limitation → feature map $\phi$ → polynomial features → polynomial kernel → kernel trick → kernel perceptron → RBF → local vs global → valid kernels → kernel ridge regression → k-NN → bias–variance in k-NN → choosing $k$ → cost & curse of dimensionality.
:::

:::add Added note
**Scope.** These notes integrate the professor's Chapter 3 Revised PDF and the derivations developed in class. Soft-margin SVM, KKT, Lasso, and Elastic Net are intentionally outside this part.
:::

## 3.3.1 Why Linear Models Need Help

A linear classifier such as the perceptron or SVM uses a hyperplane

$$\mathbf{w}^T\mathbf{x}+b=0$$

while ordinary least squares uses a linear prediction rule. These models work well when the relevant geometry is approximately linear, but many datasets have structure that **cannot be captured by one line or hyperplane**.

The enclosed-class example is worth setting up carefully. Suppose a sample is

$$\mathbf{x}=\begin{bmatrix}x_1\\x_2\end{bmatrix}\in\mathbb{R}^2.$$

The inner class lies near the origin while the outer class forms a ring. A straight line divides the plane into only two half-spaces, so no single line can put the entire inner class on one side and the entire ring on the other.

### A useful feature can change the geometry

A natural feature for this geometry is the squared distance from the origin:

$$x_1^2+x_2^2.$$

Define

$$\boxed{\phi(\mathbf{x})=\begin{bmatrix}x_1\\x_2\\x_1^2+x_2^2\end{bmatrix}}\qquad\text{so that}\qquad\mathbb{R}^2\to\mathbb{R}^3.$$

Points near the center have a small third coordinate; points on the outer ring have a larger third coordinate. In the lifted space, a **flat plane** can separate them.

:::fig lift
:::

:::def Key result
A boundary can be **nonlinear in the original input coordinates** while still being a **linear hyperplane in the transformed features**. For example, a plane involving the third feature $x_1^2+x_2^2$ maps back to a circular boundary in the original 2D space.
:::

E.g. the plane $x_1^2+x_2^2=2.1$ in the lifted space is exactly the circle of radius $\sqrt{2.1}\approx1.45$ drawn dashed on the left.

:::viz lift | Drag to orbit. Each point is lifted to height $x_1^2+x_2^2$; move the flat plane up and down and watch the matching circle on the floor.
:::

:::ml Why this matters
The goal is not "higher dimensions for their own sake." The goal is to find a **representation** in which the structure becomes easier for a linear algorithm to separate or fit.
:::

:::add Why not simply lift each class to a different height?
For the training set, one could cheat and define a new coordinate using the label itself, for example

$$\tilde\phi(\mathbf{x})=\begin{bmatrix}x_1\\x_2\\y\end{bmatrix}.$$

Then class 0 and class 1 would sit at different heights and a plane would separate them perfectly. But this is **not a valid predictive feature map**, because for a new test sample the label $y$ is exactly what we are trying to predict. A legitimate feature map must be **computable from the input alone**. The circular mapping

$$\phi(\mathbf{x})=\begin{bmatrix}x_1\\x_2\\x_1^2+x_2^2\end{bmatrix}$$

is valid because $x_1^2+x_2^2$ can be computed for any unseen point without knowing its class. (Same rule as §2.3.1: nothing the test set hides may leak into the model.)
:::

## 3.3.2 Nonlinear Feature Mappings

In general, define a mapping

$$\phi:\mathbb{R}^{d_1}\to\mathbb{R}^{d_2},\qquad d_2>d_1.$$

A linear classifier in feature space is

$$\boxed{h_\phi(\mathbf{x})=\operatorname{sign}\big(\mathbf{w}^T\phi(\mathbf{x})+b\big)}$$

The mapping must be **meaningful**: moving to a larger space is useful only if the transformed coordinates expose a structure that a linear model can exploit. For the circular example, $\phi(\mathbf{x})=[x_1,\ x_2,\ x_1^2+x_2^2]^T$ is useful because **radius** is exactly the property distinguishing the inner and outer classes.

:::add Intuition
A feature map is like **changing the camera angle** on the same dataset. Nothing about the labels changes; only the representation changes. From the right angle, a pattern that looked tangled can become linearly separable.
:::

## 3.3.3 Polynomial Features Before Polynomial Kernels

Before defining a polynomial kernel, first understand polynomial *features*. For $\mathbf{x}=[x_1,\ x_2]^T$, a linear model only sees $x_1$ and $x_2$. If we add polynomial features, the model may also use

$$1,\quad x_1,\quad x_2,\quad x_1x_2,\quad x_1^2,\quad x_2^2,\quad\dots$$

A model that is linear in these transformed features can represent a **curved boundary** in the original coordinates.

### What are $\mathbf{x}$ and $\mathbf{x}'$?

Kernel formulas compare **two ordinary input samples**. We will write

$$\mathbf{x}=\begin{bmatrix}x_1\\x_2\end{bmatrix},\qquad\mathbf{x}'=\begin{bmatrix}x_1'\\x_2'\end{bmatrix}.$$

The prime in $\mathbf{x}'$ does **not** mean transformed. It simply means "a second sample." The transformed versions are $\phi(\mathbf{x})$ and $\phi(\mathbf{x}')$.

:::add Clarification
Keep these symbols separate:

$$\begin{aligned}\mathbf{x}&=\text{first original sample},\\\mathbf{x}'&=\text{second original sample},\\\phi(\mathbf{x})&=\text{transformed version of }\mathbf{x}.\end{aligned}$$
:::

## 3.3.4 The Polynomial Kernel, Built from First Principles

The degree-$d$ polynomial kernel used in the lecture is

:::def Polynomial kernel
$$K(\mathbf{x},\mathbf{x}')=(1+\mathbf{x}^T\mathbf{x}')^d$$
:::

For $d=2$: $\ K(\mathbf{x},\mathbf{x}')=(1+\mathbf{x}^T\mathbf{x}')^2$.

### Why is there a $1+$?

Without the constant, $(\mathbf{x}^T\mathbf{x}')^2$ contains only degree-2 interactions. Adding 1 gives $(1+\mathbf{x}^T\mathbf{x}')^2$, which produces degree 0, degree 1, and degree 2 terms. Thus the 1 allows **lower-order terms and a constant/bias feature** to be included.

:::def Key result
$$(\mathbf{x}^T\mathbf{x}')^d\;\Rightarrow\;\text{degree }d\text{ terms only},\qquad(1+\mathbf{x}^T\mathbf{x}')^d\;\Rightarrow\;\text{degrees }0,1,\dots,d.$$
:::

### Exact $n=2$, $d=2$ derivation

Since $\mathbf{x}^T\mathbf{x}'=x_1x_1'+x_2x_2'$, we have

$$\begin{aligned}K(\mathbf{x},\mathbf{x}')&=(1+x_1x_1'+x_2x_2')^2\\&=1+2x_1x_1'+2x_2x_2'+x_1^2(x_1')^2+2x_1x_2x_1'x_2'+x_2^2(x_2')^2.\end{aligned}$$

Now ask a very specific question:

:::cap
*Can this whole expression be written as an ordinary dot product between two transformed vectors?*
:::

Match each kernel term with a product of one feature from $\mathbf{x}$ and the corresponding feature from $\mathbf{x}'$:

$$1=(1)(1),\qquad2x_1x_1'=(\sqrt2\,x_1)(\sqrt2\,x_1'),\qquad2x_1x_2x_1'x_2'=(\sqrt2\,x_1x_2)(\sqrt2\,x_1'x_2'),$$

and similarly for the remaining terms. This produces

$$\phi(\mathbf{x})=\begin{bmatrix}1\\\sqrt2\,x_1\\\sqrt2\,x_2\\\sqrt2\,x_1x_2\\x_1^2\\x_2^2\end{bmatrix},\qquad\phi(\mathbf{x}')=\begin{bmatrix}1\\\sqrt2\,x_1'\\\sqrt2\,x_2'\\\sqrt2\,x_1'x_2'\\(x_1')^2\\(x_2')^2\end{bmatrix}.$$

Therefore

$$\begin{aligned}\phi(\mathbf{x})^T\phi(\mathbf{x}')&=(1)(1)+(\sqrt2x_1)(\sqrt2x_1')+(\sqrt2x_2)(\sqrt2x_2')+(\sqrt2x_1x_2)(\sqrt2x_1'x_2')+x_1^2(x_1')^2+x_2^2(x_2')^2\\&=1+2x_1x_1'+2x_2x_2'+2x_1x_2x_1'x_2'+x_1^2(x_1')^2+x_2^2(x_2')^2\\&=(1+\mathbf{x}^T\mathbf{x}')^2.\end{aligned}$$

Hence

$$\boxed{K(\mathbf{x},\mathbf{x}')=\phi(\mathbf{x})^T\phi(\mathbf{x}')}$$

:::add Intuition
We could transform every 2D point into the 6D vector $\phi(\mathbf{x})$ and compute there. The kernel gives the **same high-dimensional dot product directly from the original inputs**: behave as if we entered feature space without explicitly constructing it.
:::

:::ex Check with numbers
$\mathbf{x}=(1,2)$, $\mathbf{x}'=(3,1)$: $\ \mathbf{x}^T\mathbf{x}'=3+2=5$, so $K=(1+5)^2=36$.\
The long way: $\phi(\mathbf{x})=(1,\ \sqrt2,\ 2\sqrt2,\ 2\sqrt2,\ 1,\ 4)$, $\ \phi(\mathbf{x}')=(1,\ 3\sqrt2,\ \sqrt2,\ 3\sqrt2,\ 9,\ 1)$, so

$$\phi(\mathbf{x})^T\phi(\mathbf{x}')=1+6+4+12+9+4=36.\ \checkmark$$

Same number, but the kernel needed one 2D dot product instead of building two 6D vectors.
:::

### Why explicit polynomial maps become expensive

For an input $\mathbf{x}\in\mathbb{R}^n$, the lecture gives the number of polynomial monomial features up to degree $d$ as

$$\boxed{|\phi(\mathbf{x})|=\binom{n+d}{d}=\frac{(n+d)!}{n!\,d!}}$$

(Check: $n=2$, $d=2$ gives $\binom42=6$, exactly the 6D vector above.) Examples from the lecture:

$$n=20,\ d=2\Rightarrow231,\qquad n=100,\ d=2\Rightarrow5{,}151,\qquad n=100,\ d=3\Rightarrow176{,}851.$$

:::fig feature-count
:::

Feature explosion is the motivation for computing $K(\mathbf{x},\mathbf{x}')$ directly instead of constructing every component of $\phi(\mathbf{x})$: the kernel always costs one $n$-dimensional dot product, no matter how big $\phi$ is.

:::viz polykernel | Drag $\mathbf{x}$ and $\mathbf{x}'$: the kernel and the six-term dot product always agree. Below, $n$ and $d$ set the feature count $\binom{n+d}{d}$.
:::

## 3.3.5 Why Dot Products Let Us Kernelize the Perceptron

Start with the ordinary perceptron (using learning rate $\eta=1$ for clarity). If sample $(\mathbf{x}_i,y_i)$ is misclassified,

$$y_i(\mathbf{w}^T\mathbf{x}_i+b)\le0,$$

then

$$\mathbf{w}\leftarrow\mathbf{w}+y_i\mathbf{x}_i,\qquad b\leftarrow b+y_i.$$

### Rewrite $\mathbf{w}$ as a sum of past updates

Every time training sample $j$ causes an update, the vector $y_j\mathbf{x}_j$ is added to $\mathbf{w}$. Define

$$\boxed{\alpha_j=\text{number of times sample }j\text{ caused an update}}$$

Then after training

$$\mathbf{w}=\sum_{j=1}^n\alpha_jy_j\mathbf{x}_j,\qquad b=\sum_{j=1}^n\alpha_jy_j.$$

This is **not a new model assumption**; it is a bookkeeping rewrite of the original perceptron updates. (Same shape as the SVM result $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$ from §3.1.7 — there it came out of the Lagrangian, here out of counting updates.)

Substitute the expansion of $\mathbf{w}$ into the score for sample $\mathbf{x}_i$:

$$\mathbf{w}^T\mathbf{x}_i+b=\Big(\sum_j\alpha_jy_j\mathbf{x}_j\Big)^T\mathbf{x}_i+b=\sum_j\alpha_jy_j\underbrace{\mathbf{x}_j^T\mathbf{x}_i}_{\text{pairwise dot product}}+b.$$

The crucial fact is that the algorithm can now be written using **only dot products between samples**.

### Kernelized perceptron

In feature space, the pairwise dot product would be $\phi(\mathbf{x}_j)^T\phi(\mathbf{x}_i)$. But by definition of a kernel,

$$\phi(\mathbf{x}_j)^T\phi(\mathbf{x}_i)=K(\mathbf{x}_j,\mathbf{x}_i).$$

Therefore the classifier can be written

:::def Kernel perceptron
$$f(\mathbf{x}_i)=\operatorname{sign}\Big(\sum_{j=1}^n\alpha_jy_j\,K(\mathbf{x}_j,\mathbf{x}_i)+b\Big)$$

If $\mathbf{x}_i$ is misclassified, we simply increment

$$\alpha_i\leftarrow\alpha_i+1.$$
:::

:::ex Concrete counter example
Suppose there are five samples. Initialize $\alpha_1=\cdots=\alpha_5=0$.\
During the first pass, suppose samples 1, 2, and 5 cause updates. Then $\alpha_1=1$, $\alpha_2=1$, $\alpha_5=1$, and

$$\mathbf{w}=y_1\mathbf{x}_1+y_2\mathbf{x}_2+y_5\mathbf{x}_5.$$

If sample 1 causes another update on the next pass, $\alpha_1=2$, so

$$\mathbf{w}=2y_1\mathbf{x}_1+y_2\mathbf{x}_2+y_5\mathbf{x}_5.$$

Thus $\alpha_j$ literally records how many times that training sample has contributed its update vector.
:::

:::def Key result
The reason the kernel trick can be inserted into the perceptron is not magic: after rewriting $\mathbf{w}$, the prediction depends on $\mathbf{x}_j^T\mathbf{x}_i$. Replace the transformed-space dot product $\phi(\mathbf{x}_j)^T\phi(\mathbf{x}_i)$ by $K(\mathbf{x}_j,\mathbf{x}_i)$, and the explicit feature vector disappears.
:::

This is exactly the doorway promised in the SVM dual (§3.1.10): the dual only touches $\mathbf{x}_i^T\mathbf{x}_j$, so swapping in $K(\mathbf{x}_i,\mathbf{x}_j)$ gives a **kernel SVM** the same way.

:::viz kperceptron | Run passes of the kernel perceptron on the ring data. Each number is a point's $\alpha_j$ (how many updates it caused); the colour is the sign of $\sum_j\alpha_jy_jK(\mathbf{x}_j,\mathbf{x})+b$.
:::

## 3.3.6 RBF / Gaussian Kernel

:::def RBF kernel
$$K(\mathbf{x},\mathbf{x}')=\exp\big(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2\big),\qquad\gamma>0.$$

It measures similarity using Euclidean distance (§2.2.2).
:::

If $\mathbf{x}=\mathbf{x}'$, then $\|\mathbf{x}-\mathbf{x}'\|^2=0$ and $K(\mathbf{x},\mathbf{x}')=e^0=1$. As the distance grows, the exponent becomes more negative and the similarity **decays toward zero**.

E.g. with $\gamma=1$: distance 1 gives $e^{-1}\approx0.37$, distance 2 gives $e^{-4}\approx0.018$. Doubling the distance kills the similarity almost entirely.

:::fig rbf-profiles
:::

:::cap
Small $\gamma$ gives broad influence; large $\gamma$ makes similarity decay quickly and produces very local influence.
:::

### Full derivation of the implicit infinite-dimensional map

Start from

$$K(\mathbf{x},\mathbf{x}')=e^{-\gamma\|\mathbf{x}-\mathbf{x}'\|^2}.$$

Use the vector identity

$$\|\mathbf{x}-\mathbf{x}'\|^2=\|\mathbf{x}\|^2+\|\mathbf{x}'\|^2-2\mathbf{x}^T\mathbf{x}'.$$

Therefore

$$\begin{aligned}K(\mathbf{x},\mathbf{x}')&=e^{-\gamma(\|\mathbf{x}\|^2+\|\mathbf{x}'\|^2-2\mathbf{x}^T\mathbf{x}')}\\&=e^{-\gamma\|\mathbf{x}\|^2}\,e^{-\gamma\|\mathbf{x}'\|^2}\,e^{2\gamma\mathbf{x}^T\mathbf{x}'}.\end{aligned}$$

The first two factors each depend on only one sample; all the interaction between $\mathbf{x}$ and $\mathbf{x}'$ lives in the third. Now use the Taylor/Maclaurin series

$$e^z=\sum_{k=0}^\infty\frac{z^k}{k!}=1+z+\frac{z^2}{2!}+\frac{z^3}{3!}+\cdots.$$

Set $z=2\gamma\mathbf{x}^T\mathbf{x}'$:

$$e^{2\gamma\mathbf{x}^T\mathbf{x}'}=\sum_{k=0}^\infty\frac{(2\gamma\mathbf{x}^T\mathbf{x}')^k}{k!}=1+2\gamma\mathbf{x}^T\mathbf{x}'+\frac{(2\gamma)^2(\mathbf{x}^T\mathbf{x}')^2}{2!}+\frac{(2\gamma)^3(\mathbf{x}^T\mathbf{x}')^3}{3!}+\cdots.$$

Each $(\mathbf{x}^T\mathbf{x}')^k$ is a polynomial kernel of degree $k$ (§3.3.4). So the expansion contains polynomial interactions of degree

$$0,\ 1,\ 2,\ 3,\ \dots$$

with **no finite last degree**. This is why the RBF kernel corresponds to an implicit feature representation with **infinitely many components**.

:::add Clarification
**Do we literally need infinite dimensions?** No. A particular finite dataset might have some finite clever mapping that separates it. The point of RBF is that we **do not need to guess that mapping in advance**. The kernel gives access to an extremely rich implicit feature space while still requiring only the simple quantity $e^{-\gamma\|\mathbf{x}-\mathbf{x}'\|^2}$.
:::

### Circle versus complex local patterns

For the circular dataset, one extra feature such as $x_1^2+x_2^2$ may be enough to create a clean 3D separation. For a pattern with many alternating local class regions, there may be no obvious single feature to invent. RBF avoids committing to one low-degree polynomial map and provides a rich family of nonlinear interactions automatically.

## 3.3.7 Local versus Global Geometry

The words *local* and *global* describe how an adjustment or training point influences the input space.

- **Global polynomial feature:** a feature such as $x^2$ or $x^3$ is defined everywhere. Changing its coefficient changes the fitted curve **across the whole domain**.
- **Local RBF influence:** $K(\mathbf{x},\mathbf{x}_i)=e^{-\gamma\|\mathbf{x}-\mathbf{x}_i\|^2}$ is strongest near $\mathbf{x}_i$ and fades toward zero as we move away from $\mathbf{x}_i$.

:::fig global-local
:::

:::add Intuition
Think of a polynomial model as **bending one large sheet**: changing the global shape affects distant regions too. Think of RBF as **combining many local hills**: a training point has strong influence nearby and little influence far away.
:::

This explains the role of $\gamma$:

$$\begin{gathered}\gamma\downarrow\;\Rightarrow\;\text{wide neighborhoods, smoother boundary,}\\\gamma\uparrow\;\Rightarrow\;\text{narrow neighborhoods, more detailed boundary, greater overfitting risk.}\end{gathered}$$

Same bias–variance dial as §2.3.2: small $\gamma$ leans toward underfitting, large $\gamma$ toward overfitting.

:::viz rbf | Slide $\gamma$. Left: the RBF and Laplacian similarity against distance. Right: an RBF kernel classifier on the ring data, from smooth (small $\gamma$) to very local (large $\gamma$).
:::

## 3.3.8 Polynomial versus RBF: What Each Kernel Assumes

The polynomial kernel

$$K_{\text{poly}}(\mathbf{x},\mathbf{x}')=(1+\mathbf{x}^T\mathbf{x}')^d$$

uses a **finite** set of polynomial interactions up to degree $d$ and imposes a more **global** geometry. Increasing $d$ increases flexibility, but high degree can produce unstable or oscillatory global boundaries.

The RBF kernel

$$K_{\text{RBF}}(\mathbf{x},\mathbf{x}')=e^{-\gamma\|\mathbf{x}-\mathbf{x}'\|^2}$$

provides an **infinite** expansion with **local** distance-based similarity. This makes it natural for complicated smooth patterns whose decision geometry varies from region to region.

|{stack}
| **Kernel** | Main geometric idea | Typical failure mode from the lecture |
|---|---|---|
| **Linear** | One hyperplane; no nonlinear mapping | Cannot capture curved/XOR-like structure |
| **Polynomial** | Global polynomial interactions up to degree $d$ | Small $d$ underfits; large $d$ may overfit/oscillate; poor locality |
| **RBF** | Smooth localized similarity; very rich implicit feature space | $\gamma$ too small underfits; $\gamma$ too large memorizes/local-overfits |
| **Laplacian** | Local similarity with a sharper, less smooth peak | Can become overly sharp/noise-sensitive |
| **Sigmoid** | Neural-activation-like similarity | Sensitive to parameters/scaling; not valid for all parameter choices in the lecture notes |

## 3.3.9 Exponential / Laplacian Kernel

The exponential (Laplacian) kernel uses distance **without squaring it**:

$$\boxed{K_{\text{Lap}}(\mathbf{x},\mathbf{x}')=e^{-\gamma\|\mathbf{x}-\mathbf{x}'\|}}$$

In one dimension this resembles $e^{-\gamma|x-x'|}$. The absolute value produces a **sharper peak** than the RBF's smooth Gaussian profile.

:::fig rbf-laplacian
:::

Same squared-vs-not-squared story as $L_2$ vs $L_1$ (§1.1.7) and MSE vs MAE (§2.3.8). The lecture relates the exponential kernel to the broader **Matérn family**: $\nu=\frac12$ gives the exponential form, while the RBF is recovered in the infinitely smooth limit $\nu\to\infty$.

## 3.3.10 What Makes a Kernel Valid?

A kernel must behave like an inner product in some feature space:

$$K(\mathbf{x},\mathbf{x}')=\phi(\mathbf{x})^T\phi(\mathbf{x}').$$

The lecture summarizes this through **positive semi-definiteness**: for any set of samples, the matrix of pairwise values $K_{ij}=K(\mathbf{x}_i,\mathbf{x}_j)$ must be symmetric with no negative eigenvalues.

Why that is the right test: if $K$ really is a dot product, then for any weights $\mathbf{c}$,

$$\mathbf{c}^TK\mathbf{c}=\sum_i\sum_jc_ic_j\,\phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)=\Big\|\sum_ic_i\phi(\mathbf{x}_i)\Big\|^2\ge0.$$

A squared length can never be negative, so a genuine dot product can never produce a matrix with a negative direction.

According to the lecture notes, valid kernels can be combined through operations such as **addition, positive scalar multiplication, products, and certain powers**. This permits composite kernels, for example

$$K=K_{\text{poly}}+K_{\text{RBF}},$$

which can mix different notions of similarity.

:::def Key result
Choosing a kernel means choosing a **notion of similarity**. That notion of similarity defines the geometry available to the model and therefore the kinds of decision boundaries it can form.
:::

:::viz gram | Pick a kernel and drag the six points. The heat-table is the matrix $K_{ij}=K(\mathbf{x}_i,\mathbf{x}_j)$; its smallest eigenvalue decides validity.
:::

## 3.3.11 Kernel Least Squares / Kernel Ridge Regression

Ordinary least squares (§3.2) predicts $\hat{\mathbf{y}}=X\boldsymbol\beta$ with $\hat{\boldsymbol\beta}=(X^TX)^{-1}X^T\mathbf{y}$. For nonlinear data, map each sample first:

$$\mathbf{x}_i\to\phi(\mathbf{x}_i).$$

Define the feature matrix

$$\Phi=\phi(X)=\begin{bmatrix}\phi(\mathbf{x}_1)^T\\\phi(\mathbf{x}_2)^T\\\vdots\\\phi(\mathbf{x}_n)^T\end{bmatrix}.$$

Then $\hat{\mathbf{y}}=\Phi\boldsymbol\beta$. Explicitly computing $\Phi$ can again be expensive or impossible for a huge/infinite feature map.

### The kernel matrix

Define

$$\boxed{K=\Phi\Phi^T\in\mathbb{R}^{n\times n}}\qquad\text{so that}\qquad\boxed{K_{ij}=\phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)=K(\mathbf{x}_i,\mathbf{x}_j)}$$

$K$ is a table of **pairwise similarities** between training samples.

### What does $\boldsymbol\alpha$ mean here?

In kernel regression, $\boldsymbol\alpha$ is a vector of learned weights attached to the training samples:

$$\boldsymbol\alpha=[\,\alpha_1\ \cdots\ \alpha_n\,]^T.$$

This is **not the same semantic role** as the perceptron update counter, even though the same symbol is reused. Write

$$\boxed{\boldsymbol\beta=\Phi^T\boldsymbol\alpha}$$

(again: the weight vector is a combination of training samples). Then

$$\hat{\mathbf{y}}=\Phi\boldsymbol\beta=\Phi\Phi^T\boldsymbol\alpha=K\boldsymbol\alpha.$$

Thus a high-dimensional parameter vector $\boldsymbol\beta$ is replaced by **one coefficient per training example**.

### Regularized kernel least-squares objective

The lecture uses

:::def Kernel ridge objective
$$L(\boldsymbol\alpha)=\|\mathbf{y}-K\boldsymbol\alpha\|^2+\lambda\boldsymbol\alpha^TK\boldsymbol\alpha$$
:::

The penalty is not arbitrary: it is the ridge penalty $\lambda\|\boldsymbol\beta\|^2$ from §3.2.8 rewritten in terms of $\boldsymbol\alpha$, since

$$\|\boldsymbol\beta\|^2=\boldsymbol\beta^T\boldsymbol\beta=\boldsymbol\alpha^T\Phi\Phi^T\boldsymbol\alpha=\boldsymbol\alpha^TK\boldsymbol\alpha.$$

Differentiate the first term using the same OLS pattern:

$$\frac{\partial}{\partial\boldsymbol\alpha}\|\mathbf{y}-K\boldsymbol\alpha\|^2=-2K^T(\mathbf{y}-K\boldsymbol\alpha).$$

Since a kernel matrix is symmetric, $K^T=K$, so this is $-2K(\mathbf{y}-K\boldsymbol\alpha)$. For the regularization term, use the quadratic-form identity (§1.3.5). Since $K^T=K$,

$$\frac{\partial}{\partial\boldsymbol\alpha}(\boldsymbol\alpha^TK\boldsymbol\alpha)=2K\boldsymbol\alpha,$$

so

$$\nabla_{\boldsymbol\alpha}L=-2K(\mathbf{y}-K\boldsymbol\alpha)+2\lambda K\boldsymbol\alpha.$$

Set the gradient to zero:

$$\begin{aligned}-2K(\mathbf{y}-K\boldsymbol\alpha)+2\lambda K\boldsymbol\alpha&=\mathbf{0}\\-K\mathbf{y}+K^2\boldsymbol\alpha+\lambda K\boldsymbol\alpha&=\mathbf{0}.\end{aligned}$$

Every term has a $K$ in front, so factor it out:

$$K\big(K\boldsymbol\alpha+\lambda\boldsymbol\alpha-\mathbf{y}\big)=\mathbf{0}.$$

When $K$ is invertible, multiply by $K^{-1}$ and the bracket itself must vanish. Under the regularized kernel-ridge solution used in the lecture, this yields

$$\boxed{(K+\lambda I)\boldsymbol\alpha=\mathbf{y}}$$

and therefore

$$\boxed{\hat{\boldsymbol\alpha}=(K+\lambda I)^{-1}\mathbf{y}}\qquad\qquad\boxed{\hat{\mathbf{y}}=K(K+\lambda I)^{-1}\mathbf{y}}$$

Compare ridge in §3.2.8: $(X^TX+\lambda I)^{-1}$ is $p\times p$ (one row per *feature*); here $(K+\lambda I)^{-1}$ is $n\times n$ (one row per *sample*). That swap is the whole reason the feature space can be infinite.

### Prediction at a new point

For a new input $\mathbf{x}_*$,

:::def Kernel regression prediction
$$\hat y(\mathbf{x}_*)=\sum_{i=1}^n\alpha_i\,K(\mathbf{x}_i,\mathbf{x}_*)$$
:::

Interpretation:

- $\alpha_i$ says how strongly training sample $i$ contributes.
- $K(\mathbf{x}_i,\mathbf{x}_*)$ says how similar the new point is to training sample $i$.
- The prediction is the **weighted sum of these similarity contributions**.

:::fig krr-fit
:::

:::ml Why this matters
For RBF regression, nearby training points have large kernel values and distant points have values near zero. The model can therefore **follow local curvature**, unlike a single global straight OLS line.
:::

:::viz krr | Set $\gamma$ and $\lambda$. The thin curves are the terms $\alpha_iK(\mathbf{x}_i,x)$; their sum is the kernel ridge prediction. The straight line is OLS on the same points.
:::

### Kernel-regression limitations

The lecture emphasizes:

- $K$ is $n\times n$, so storage is $O(n^2)$.
- Matrix inversion is approximately $O(n^3)$ in the basic formulation.
- Flexible kernels can overfit.
- Performance is sensitive to hyperparameters such as $\gamma$, polynomial degree $d$, and regularization $\lambda$.
- Similarity can become less discriminative in high dimensions (§2.2.3).

## 3.3.12 k-Nearest Neighbors (k-NN)

k-NN is conceptually different from the preceding models. It does **not** learn an explicit $\mathbf{w}$, $\boldsymbol\beta$, or kernel coefficient function during a conventional training stage. It **stores the training set** and postpones most work until a query arrives. This is why it is called **instance-based** or **lazy learning**.

For a query point $\mathbf{x}_q$:

1. compute its distance to training samples;
2. identify the $k$ closest samples;
3. classify by vote or regress by averaging their target values.

:::fig knn-star
:::

### Distance metrics

These are the distances from §2.2.2. Euclidean distance:

$$d(\mathbf{x}_q,\mathbf{x}_i)=\sqrt{\sum_{j=1}^d(x_{qj}-x_{ij})^2}.$$

Manhattan distance:

$$d(\mathbf{x}_q,\mathbf{x}_i)=\sum_{j=1}^d|x_{qj}-x_{ij}|.$$

The **Minkowski** family combines both:

$$\boxed{d_p(\mathbf{x}_q,\mathbf{x}_i)=\Big(\sum_j|x_{qj}-x_{ij}|^p\Big)^{1/p}}$$

with $p=1$ giving Manhattan and $p=2$ giving Euclidean distance.

### Why feature scaling is critical

Distance is numerical. If one feature ranges from 0 to 200,000 and another from 0 to 100, the first feature can dominate the distance simply because of its units. Standard fixes (§2.1.4) are min–max normalization

$$x'=\frac{x-\min(x)}{\max(x)-\min(x)}$$

or standardization

$$x'=\frac{x-\mu}{\sigma}.$$

:::def Key result
In k-NN, "nearest" is meaningful **only after the feature scales make sense**. Otherwise the largest-scale feature may silently define the neighborhood.
:::

### Classification, ties, and weighted voting

For classification,

$$\boxed{\hat y_q=\operatorname{mode}(y_1,\dots,y_k)}$$

Example: $[0,1,1,0,1]\Rightarrow\hat y_q=1$.

A tie such as $[0,1,1,0]$ needs a tie-breaking rule. The lecture lists random selection, preference for the single nearest neighbor, and **weighted voting**. A common distance weight is

$$\boxed{w_i=\frac{1}{d(\mathbf{x}_q,\mathbf{x}_i)+\epsilon}}$$

where $\epsilon$ prevents division by zero. Closer points receive larger voting influence.

:::ex Weighting can flip the vote
$k=3$ neighbors at distances $1,2,4$ with labels $1,0,0$.\
Plain vote: two 0's vs one $1\Rightarrow$ predict 0.\
Weighted: $w=1,\ 0.5,\ 0.25$. Class 1 gets 1; class 0 gets $0.5+0.25=0.75\Rightarrow$ predict **1**. The one very close neighbor outweighs two far ones.
:::

:::viz knn | Drag the query $\mathbf{x}_q$ and change $k$. The dashed circle reaches the $k$-th neighbour; the shading is the prediction everywhere for this $k$. Switch on distance weighting to compare.
:::

### k-NN regression

For regression, average the targets of the $k$ nearest samples:

$$\hat y_q=\frac1k\sum_{i=1}^ky_i.$$

Or use a weighted average:

$$\hat y_q=\frac{\sum_{i=1}^kw_iy_i}{\sum_{i=1}^kw_i}.$$

This weighted average is kernel-regression-flavoured: closer samples count more, just with a hard cutoff at $k$ neighbors instead of a smooth decay.

## 3.3.13 The Meaning of Bias and Variance in k-NN

The parameter $k$ controls how local or smooth the model is. This is a classic bias–variance tradeoff (§2.3.2).

### Variance: sensitivity to the exact training sample

**Variance** means how strongly the learned predictions would change if the training dataset changed slightly. For $k=1$, the entire prediction depends on one nearest point. Remove that point, move it slightly, or replace it with a noisy/mislabeled point, and the prediction may change dramatically. Therefore

$$\boxed{k\text{ small}\Rightarrow\text{high variance}}$$

This is why small $k$ can overfit noise.

### Bias: systematic error from too much smoothing

**Bias** means systematic error caused by a model being too rigid or too smooth to represent real structure. For large $k$, predictions average over a broad neighborhood. Local bumps, class pockets, and genuine small-scale structure are washed out. In the extreme $k\to n$, regression approaches the **global average** target and classification approaches the **globally dominant class**. Thus

$$\boxed{k\text{ large}\Rightarrow\text{high bias}}$$

:::fig knn-k
:::

:::def Key result
$$\begin{gathered}k\text{ small}\Rightarrow\text{low bias, high variance (overfitting risk)}\\k\text{ large}\Rightarrow\text{high bias, low variance (underfitting risk)}\end{gathered}$$
:::

## 3.3.14 Choosing $k$: the Elbow / Minimum-Error Region

The lecture plots validation error or regression MSE as a function of $k$.

:::fig elbow
:::

Read the graph from left to right:

1. At very small $k$, error is high because the model has high variance and follows local noise.
2. As $k$ increases, averaging reduces noise and MSE falls.
3. Around $k\approx10$–11, this example reaches a good bias–variance balance.
4. Increasing $k$ further makes the model too smooth; bias rises and error begins to increase.

:::add Clarification
The word "elbow" refers to the region where improvement starts flattening or reversing. In this lecture figure, the practically important feature is the **minimum-error region** around $k\approx10$–11; $k$ should be chosen from **validation performance**, not by habit (K-fold cross-validation, §2.3.3, is the standard way).
:::

:::viz knnreg | Move $k$: k-NN regression on noisy data, from $k=1$ (follows every point) to large $k$ (washes the structure out). The right plot is the real validation MSE for every $k$ on held-out points.
:::

## 3.3.15 k-NN Computational Cost and the Curse of Dimensionality

For one query, brute-force k-NN computes distance to all $n$ training samples. With $d$ features,

$$\boxed{\text{query time}=O(nd)}\qquad\qquad\boxed{\text{memory}=O(nd)}$$

(memory because the entire dataset is stored). Note the flip compared with the other models: training is free, prediction is expensive. The lecture lists **KD-trees** for low-dimensional data, **ball trees** for moderate dimensions, and **approximate-nearest-neighbor** methods for large-scale search.

### Curse of dimensionality

As dimension grows, Euclidean distances tend to become less distinguishable (§2.2.3). A nearest point, an average point, and a farthest point can all become relatively similar in distance.

:::fig curse-sim
:::

:::cap
Simulation: 1000 uniform points, one random query. As $d$ grows, nearest and farthest distances squeeze toward the average.
:::

This hurts k-NN because its entire logic depends on the nearest neighbors being **meaningfully nearer** than the rest. The lecture suggests dimensionality reduction such as PCA/LDA or feature selection (§2.1.7–2.1.9) before k-NN when needed.

### Class imbalance

If one class dominates the dataset, the local neighborhood can also be dominated by that class. Distance-weighted voting or class-specific weighting can reduce this effect (and see the sampling fixes in §2.1.3).

## 3.3.16 Quick Reference: One-Page Mental Model

|{stack}
| **Concept** | Core idea | Formula / cue |
|---|---|---|
| **Feature mapping** | Re-express data so a linear model can solve a nonlinear-looking problem | $\mathbf{x}\mapsto\phi(\mathbf{x})$ |
| **Polynomial kernel** | Dot product in a finite polynomial feature space without building it | $(1+\mathbf{x}^T\mathbf{x}')^d$ |
| **Feature count** | Explicit polynomial maps explode | $\binom{n+d}{d}$ |
| **Kernel trick** | Replace $\phi(\mathbf{x})^T\phi(\mathbf{x}')$ with direct $K(\mathbf{x},\mathbf{x}')$ | "high-dimensional dot product for free" |
| **Kernel perceptron** | Rewrite $\mathbf{w}$ as weighted training samples, then kernelize the dot products | $\sum_j\alpha_jy_jK(\mathbf{x}_j,\mathbf{x})$ |
| **RBF** | Local distance-based similarity with an infinite polynomial expansion | $e^{-\gamma\|\mathbf{x}-\mathbf{x}'\|^2}$ |
| $\gamma$ | Width of each local hill | $\gamma\uparrow$: narrow, detailed, overfit risk |
| **Laplacian** | Sharper local distance similarity | $e^{-\gamma\|\mathbf{x}-\mathbf{x}'\|}$ |
| **Valid kernel** | Behaves like a dot product | $K$ symmetric PSD |
| **Kernel ridge** | One weight per training sample | $\hat{\boldsymbol\alpha}=(K+\lambda I)^{-1}\mathbf{y}$ |
| **Kernel regression** | Predict from weighted kernel similarities to training samples | $\hat y(\mathbf{x}_*)=\sum_i\alpha_iK(\mathbf{x}_i,\mathbf{x}_*)$ |
| **k-NN** | No explicit fitted model; vote/average the nearest stored examples | distance $\to$ $k$ neighbors $\to$ vote/mean |
| **Minkowski** | One family for both distances | $p=1$ Manhattan, $p=2$ Euclidean |
| **Weighted vote** | Closer neighbors count more | $w_i=1/(d+\epsilon)$ |
| **Bias–variance in k-NN** | $k$ controls locality versus smoothing | small $k$: high variance; large $k$: high bias |
| **Cost** | Lazy: free training, expensive queries | query $O(nd)$, memory $O(nd)$ |

:::def Final conceptual bridge
Across the whole section, the recurring question is how **similarity and representation control geometry**. Feature maps change the representation; kernels compute transformed-space similarity implicitly; k-NN skips a parametric model and uses distance directly. The mathematics differs, but the common theme is deciding **which points should look similar** and what structure that similarity makes learnable.
:::
