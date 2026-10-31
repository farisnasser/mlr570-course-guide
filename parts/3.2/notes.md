:::plain
**The flow:** linear model → residual sum of squares → OLS closed form → projection onto $\operatorname{Col}(X)$ → normal equations from geometry → homoscedasticity / heteroscedasticity → WLS → multicollinearity → Ridge regularization.
:::

:::add Added note
**Scope for this lecture.** These notes cover OLS, WLS, and Ridge Regression. Lasso and Elastic Net are intentionally omitted because they were marked as skipped material.
:::

## 3.2.1 Ordinary Least Squares (OLS): What Are We Solving?

In regression the target is **continuous**. The linear model is

$$\boxed{\mathbf{y}=X\boldsymbol\beta+\boldsymbol\epsilon}$$

with $X\in\mathbb{R}^{n\times p}$, $\boldsymbol\beta\in\mathbb{R}^p$, $\mathbf{y}\in\mathbb{R}^n$, and noise $\boldsymbol\epsilon\in\mathbb{R}^n$.

The prediction vector is

$$\hat{\mathbf{y}}=X\boldsymbol\beta.$$

For sample $i$, the **residual** is

$$e_i=y_i-\hat y_i=y_i-\mathbf{x}_i^T\boldsymbol\beta.$$

OLS chooses $\boldsymbol\beta$ to minimize the **residual sum of squares (RSS)**:

:::def OLS objective
$$L(\boldsymbol\beta)=\|\mathbf{y}-X\boldsymbol\beta\|_2^2=\sum_{i=1}^n(y_i-\mathbf{x}_i^T\boldsymbol\beta)^2$$
:::

This is exactly the squared-residual loss from the identities table (§1.3.5), and RSS$/n$ is the MSE from §2.3.8.

:::ml Why square the residuals?
If we simply summed residuals, positive and negative mistakes could cancel. Squaring makes every error nonnegative and penalizes large errors more strongly.
:::

### From the norm to an expression we can differentiate

Use $\|\mathbf{v}\|^2=\mathbf{v}^T\mathbf{v}$:

$$L(\boldsymbol\beta)=(\mathbf{y}-X\boldsymbol\beta)^T(\mathbf{y}-X\boldsymbol\beta).$$

Expand just like ordinary algebra:

$$L(\boldsymbol\beta)=\mathbf{y}^T\mathbf{y}-\mathbf{y}^TX\boldsymbol\beta-\boldsymbol\beta^TX^T\mathbf{y}+\boldsymbol\beta^TX^TX\boldsymbol\beta.$$

The two middle terms are equal because each is a scalar:

$$\mathbf{y}^TX\boldsymbol\beta=\big(\mathbf{y}^TX\boldsymbol\beta\big)^T=\boldsymbol\beta^TX^T\mathbf{y}.$$

Therefore

$$\boxed{L(\boldsymbol\beta)=\mathbf{y}^T\mathbf{y}-2\boldsymbol\beta^TX^T\mathbf{y}+\boldsymbol\beta^TX^TX\boldsymbol\beta}$$

:::add Added note
**Fast way to see that $\mathbf{y}^TX\boldsymbol\beta$ is a scalar.** Read it as $\mathbf{y}^T(X\boldsymbol\beta)$. The quantity $X\boldsymbol\beta$ is a vector in $\mathbb{R}^n$, so $\mathbf{y}^T(X\boldsymbol\beta)$ is a vector dot product. A vector-transpose times a vector gives one number.
:::

## 3.2.2 Deriving the OLS Gradient from First Principles

Differentiate the expanded loss term by term.

### Term 1: $\mathbf{y}^T\mathbf{y}$

$\mathbf{y}$ does not depend on $\boldsymbol\beta$, so

$$\frac{\partial}{\partial\boldsymbol\beta}(\mathbf{y}^T\mathbf{y})=\mathbf{0}.$$

### Term 2: $-2\boldsymbol\beta^TX^T\mathbf{y}$

Since $X^T\mathbf{y}$ is constant with respect to $\boldsymbol\beta$ (this is the $\mathbf{a}^T\mathbf{x}$ identity from §1.3.5),

$$\frac{\partial}{\partial\boldsymbol\beta}(\boldsymbol\beta^T\mathbf{a})=\mathbf{a}\qquad\Longrightarrow\qquad\frac{\partial}{\partial\boldsymbol\beta}(-2\boldsymbol\beta^TX^T\mathbf{y})=-2X^T\mathbf{y}.$$

### Term 3: the quadratic form $\boldsymbol\beta^TA\boldsymbol\beta$

A useful matrix-calculus identity (the quadratic form from §1.3.5) is

$$\frac{\partial}{\partial\boldsymbol\beta}(\boldsymbol\beta^TA\boldsymbol\beta)=(A+A^T)\boldsymbol\beta.$$

Rather than memorizing it blindly, expand a two-dimensional case. Let

$$\boldsymbol\beta=\begin{bmatrix}\beta_1\\\beta_2\end{bmatrix},\qquad A=\begin{bmatrix}a&b\\c&d\end{bmatrix}.$$

Then

$$\boldsymbol\beta^TA\boldsymbol\beta=\begin{bmatrix}\beta_1&\beta_2\end{bmatrix}\begin{bmatrix}a&b\\c&d\end{bmatrix}\begin{bmatrix}\beta_1\\\beta_2\end{bmatrix}=a\beta_1^2+(b+c)\beta_1\beta_2+d\beta_2^2.$$

Ordinary partial differentiation gives

$$\nabla_{\boldsymbol\beta}(\boldsymbol\beta^TA\boldsymbol\beta)=\begin{bmatrix}2a\beta_1+(b+c)\beta_2\\(b+c)\beta_1+2d\beta_2\end{bmatrix}=(A+A^T)\boldsymbol\beta.$$

For OLS, $A=X^TX$. This matrix is **symmetric** because

$$(X^TX)^T=X^TX.$$

Hence

$$\frac{\partial}{\partial\boldsymbol\beta}(\boldsymbol\beta^TX^TX\boldsymbol\beta)=2X^TX\boldsymbol\beta.$$

Putting the three derivatives together:

$$\nabla_{\boldsymbol\beta}L=-2X^T\mathbf{y}+2X^TX\boldsymbol\beta=-2X^T(\mathbf{y}-X\boldsymbol\beta).$$

Therefore

$$\boxed{\frac{\partial L}{\partial\boldsymbol\beta}=-2X^T(\mathbf{y}-X\boldsymbol\beta)}$$

which is exactly the least-squares gradient $2A^T(A\mathbf{x}-\mathbf{b})$ from §1.3.5 with $A\to X$, $\mathbf{b}\to\mathbf{y}$.

### Set the gradient to zero

At the least-squares minimum,

$$-2X^T(\mathbf{y}-X\boldsymbol\beta)=\mathbf{0}.$$

Divide by $-2$ and distribute $X^T$:

$$\begin{aligned}X^T(\mathbf{y}-X\boldsymbol\beta)&=\mathbf{0}\\X^T\mathbf{y}-X^TX\boldsymbol\beta&=\mathbf{0}\\X^TX\boldsymbol\beta&=X^T\mathbf{y}.\end{aligned}$$

:::def Normal equations
$$X^TX\boldsymbol\beta=X^T\mathbf{y}$$

If $X^TX$ is invertible, multiply by $(X^TX)^{-1}$ on the left. Since $(X^TX)^{-1}(X^TX)=I$,

$$\boxed{\hat{\boldsymbol\beta}_{\text{OLS}}=(X^TX)^{-1}X^T\mathbf{y}}$$
:::

The loss is a convex bowl in $\boldsymbol\beta$ (its Hessian $2X^TX$ is positive semidefinite), so this flat point is the global minimum (§1.3.6), not just a stationary point.

:::add Added note
**Why did $X^TX$ not just "cancel" earlier?** Matrices are not ordinary fractions. Cancellation is justified only by multiplying with the appropriate inverse (§1.1.1). The matrix analogue of $\frac1aa=1$ is $A^{-1}A=I$.
:::

## 3.2.3 The Geometry of OLS: the Normal Equations Come from the Picture

For a single predictor with an intercept,

$$\hat{\mathbf{y}}=\beta_0\mathbf{1}+\beta_1\mathbf{x}.$$

This means every possible prediction vector is a linear combination of the two vectors $\mathbf{1}$ and $\mathbf{x}$.

:::def Key geometric statement
Because $\hat{\mathbf{y}}=\beta_0\mathbf{1}+\beta_1\mathbf{x}$, the fitted vector $\hat{\mathbf{y}}$ must lie in

$$\operatorname{Col}(X)=\operatorname{span}\{\mathbf{1},\mathbf{x}\}.$$

The true target vector $\mathbf{y}$ generally does **not** lie in that plane — this is exactly the "$\mathbf{b}\notin\operatorname{Col}(A)$, no exact solution" case from §1.1.2. OLS therefore chooses the point $\hat{\mathbf{y}}$ in the column space that is **closest** to $\mathbf{y}$.
:::

The shortest route from a point to a plane is **perpendicular** to the plane. Therefore the residual

$$\mathbf{e}=\mathbf{y}-\hat{\mathbf{y}}$$

is perpendicular to the entire column space.

:::fig projection
:::

Because the plane is generated by $\mathbf{1}$ and $\mathbf{x}$, perpendicular to the plane means perpendicular to **both** generating vectors:

$$\boxed{(\mathbf{y}-\hat{\mathbf{y}})\cdot\mathbf{1}=0}\qquad\qquad\boxed{(\mathbf{y}-\hat{\mathbf{y}})\cdot\mathbf{x}=0}$$

Equivalently, $\mathbf{1}^T(\mathbf{y}-\hat{\mathbf{y}})=0$ and $\mathbf{x}^T(\mathbf{y}-\hat{\mathbf{y}})=0$.

### Derive the normal equations directly from the geometry

Substitute $\hat{\mathbf{y}}=\beta_0\mathbf{1}+\beta_1\mathbf{x}$.

First orthogonality equation:

$$\begin{aligned}\mathbf{1}^T(\mathbf{y}-\beta_0\mathbf{1}-\beta_1\mathbf{x})&=0\\\mathbf{1}^T\mathbf{y}-\beta_0\mathbf{1}^T\mathbf{1}-\beta_1\mathbf{1}^T\mathbf{x}&=0\\(\mathbf{1}^T\mathbf{1})\beta_0+(\mathbf{1}^T\mathbf{x})\beta_1&=\mathbf{1}^T\mathbf{y}.\end{aligned}$$

Second orthogonality equation:

$$\begin{aligned}\mathbf{x}^T(\mathbf{y}-\beta_0\mathbf{1}-\beta_1\mathbf{x})&=0\\(\mathbf{x}^T\mathbf{1})\beta_0+(\mathbf{x}^T\mathbf{x})\beta_1&=\mathbf{x}^T\mathbf{y}.\end{aligned}$$

Stack the two equations:

$$\begin{bmatrix}\mathbf{1}^T\mathbf{1}&\mathbf{1}^T\mathbf{x}\\\mathbf{x}^T\mathbf{1}&\mathbf{x}^T\mathbf{x}\end{bmatrix}\begin{bmatrix}\beta_0\\\beta_1\end{bmatrix}=\begin{bmatrix}\mathbf{1}^T\mathbf{y}\\\mathbf{x}^T\mathbf{y}\end{bmatrix}.$$

Now define

$$X=\begin{bmatrix}\mathbf{1}&\mathbf{x}\end{bmatrix},\qquad\boldsymbol\beta=\begin{bmatrix}\beta_0\\\beta_1\end{bmatrix}.$$

Then the entire system collapses to

$$\boxed{X^TX\boldsymbol\beta=X^T\mathbf{y}}$$

:::def The important connection
Calculus gave $X^T(\mathbf{y}-X\hat{\boldsymbol\beta})=\mathbf{0}$. Geometry gives exactly the same equation because $\mathbf{y}-X\hat{\boldsymbol\beta}$ is perpendicular to every column of $X$. These are not two different facts; they are the **same least-squares condition written in two languages**.
:::

### Projection matrix viewpoint

Using $\hat{\boldsymbol\beta}=(X^TX)^{-1}X^T\mathbf{y}$,

$$\hat{\mathbf{y}}=X\hat{\boldsymbol\beta}=X(X^TX)^{-1}X^T\mathbf{y}.$$

Define

$$\boxed{P_X=X(X^TX)^{-1}X^T}$$

Then $\hat{\mathbf{y}}=P_X\mathbf{y}$, so $P_X$ is the matrix that **projects $\mathbf{y}$** onto $\operatorname{Col}(X)$.

## 3.2.4 OLS Numerical Example from the Lecture

:::ex Example
Take

$$X=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},\qquad\mathbf{y}=\begin{bmatrix}1\\2\\2\end{bmatrix}.$$

Then

$$X^TX=\begin{bmatrix}3&6\\6&14\end{bmatrix},\qquad X^T\mathbf{y}=\begin{bmatrix}5\\11\end{bmatrix}$$

$$\hat{\boldsymbol\beta}=(X^TX)^{-1}X^T\mathbf{y}=\frac16\begin{bmatrix}14&-6\\-6&3\end{bmatrix}\begin{bmatrix}5\\11\end{bmatrix}\approx\begin{bmatrix}0.6667\\0.5\end{bmatrix}.$$

Hence the fitted line is

$$\boxed{\hat y=0.6667+0.5x}$$
:::

:::viz projection | Drag to orbit the left view; drag $\boldsymbol\beta$ on the right. Every $\boldsymbol\beta$ gives a point $X\boldsymbol\beta$ in the plane $\operatorname{Col}(X)$; the RSS is the squared length of the residual, smallest where the residual drops straight down onto the plane.
:::

## 3.2.5 Homoscedasticity vs. Heteroscedasticity

OLS assumes that the residual variance is **constant** across observations:

$$\boxed{\operatorname{Var}(\epsilon_i)=\sigma^2}$$

This is **homoscedasticity**. If the error spread changes with $x$, the data are **heteroscedastic**.

:::fig homo-hetero
:::

A **residual plot** (residuals vs $x$) makes the difference especially clear: a constant band vs a funnel that widens.

:::fig residual-plots
:::

:::ml Mental model
Homoscedasticity means "the model is about equally uncertain everywhere." Heteroscedasticity means "some regions of the data are much noisier than others."
:::

## 3.2.6 Weighted Least Squares (WLS)

If some observations are noisier than others, OLS still treats every squared residual equally. WLS changes the objective so **more reliable observations can matter more**.

Instead of $\sum_ie_i^2$, WLS minimizes

:::def WLS objective
$$\sum_{i=1}^nw_ie_i^2=\sum_{i=1}^nw_i(y_i-\mathbf{x}_i^T\boldsymbol\beta)^2$$

A standard choice is

$$w_i\propto\frac{1}{\sigma_i^2},$$

so large variance $\Rightarrow$ small weight, and small variance $\Rightarrow$ large weight.
:::

Define the diagonal weight matrix (§1.1.4: each residual handled on its own, nothing mixed)

$$W=\operatorname{diag}(w_1,w_2,\dots,w_n).$$

Then the loss becomes

$$\boxed{L(\boldsymbol\beta)=(\mathbf{y}-X\boldsymbol\beta)^TW(\mathbf{y}-X\boldsymbol\beta)}$$

### Why is $W$ in the middle?

Let

$$\mathbf{e}=\mathbf{y}-X\boldsymbol\beta=\begin{bmatrix}e_1\\e_2\\e_3\end{bmatrix},\qquad W=\begin{bmatrix}w_1&0&0\\0&w_2&0\\0&0&w_3\end{bmatrix}.$$

Then

$$W\mathbf{e}=\begin{bmatrix}w_1e_1\\w_2e_2\\w_3e_3\end{bmatrix}\qquad\text{and therefore}\qquad\mathbf{e}^TW\mathbf{e}=\begin{bmatrix}e_1&e_2&e_3\end{bmatrix}\begin{bmatrix}w_1e_1\\w_2e_2\\w_3e_3\end{bmatrix}=w_1e_1^2+w_2e_2^2+w_3e_3^2.$$

So the middle matrix **scales each residual** before the final dot product. (Same mechanics as $\mathbf{x}^TA\mathbf{x}$ in §1.1.7: the matrix in the middle decides how much each direction costs.)

:::def OLS is a special case of WLS
If every observation has the same weight, $W=I$, then $\mathbf{e}^TW\mathbf{e}=\mathbf{e}^T\mathbf{e}$, which is ordinary least squares.
:::

### WLS derivation

Differentiate (same shape as the OLS gradient, with $W$ riding along):

$$\frac{\partial L}{\partial\boldsymbol\beta}=-2X^TW(\mathbf{y}-X\boldsymbol\beta).$$

Set to zero:

$$\begin{aligned}X^TW(\mathbf{y}-X\boldsymbol\beta)&=\mathbf{0}\\X^TW\mathbf{y}-X^TWX\boldsymbol\beta&=\mathbf{0}\\X^TWX\boldsymbol\beta&=X^TW\mathbf{y}.\end{aligned}$$

If $X^TWX$ is invertible,

$$\boxed{\hat{\boldsymbol\beta}_{\text{WLS}}=(X^TWX)^{-1}X^TW\mathbf{y}}$$

In the lecture example with strongly non-constant variance, WLS changes the fitted line because high-variance observations are given less influence.

:::viz wls | Heteroscedastic data: the spread grows with $x$. Blend the weights from $W=I$ (OLS) to $w_i\propto1/\sigma_i^2$ (WLS) and watch the line stop chasing the noisy end; the dot size shows each weight.
:::

## 3.2.7 Ridge Regression: a Different Problem

WLS addressed unequal noise **across observations**. Ridge addresses a different issue: **multicollinearity**, where predictor **columns** of $X$ are strongly correlated.

Suppose two feature columns are almost the same:

$$\mathbf{x}_1\approx\mathbf{x}_2.$$

Then

$$\hat{\mathbf{y}}=\beta_1\mathbf{x}_1+\beta_2\mathbf{x}_2.$$

If they were exactly equal, $\mathbf{x}_1=\mathbf{x}_2=\mathbf{x}$, then

$$\hat{\mathbf{y}}=(\beta_1+\beta_2)\mathbf{x}.$$

The model can identify the **sum** $\beta_1+\beta_2$, but not how that total should be split between the two coefficients. Many coefficient pairs can produce almost the same predictions.

:::ex Concrete intuition
If $\mathbf{x}_1=\mathbf{x}_2=\mathbf{x}$, then $(\beta_1,\beta_2)=(1,0)$ and $(100,-99)$ both give $\hat{\mathbf{y}}=\mathbf{x}$. OLS can therefore become numerically unstable and assign large coefficients with opposite signs even though the predictions barely change.
:::

The OLS formula contains

$$(X^TX)^{-1}.$$

Highly correlated columns make $X^TX$ **nearly singular** — this is the redundancy-lowers-rank story from §1.1.2, happening to $X^TX$ — so its inverse becomes unstable. Tiny changes in the data can then cause large swings in $\hat{\boldsymbol\beta}$.

## 3.2.8 Ridge Objective and Closed-Form Solution

Ridge keeps the least-squares fit term but adds a penalty on coefficient magnitude — the $L_2$ regularization from §1.1.7:

:::def Ridge objective
$$L(\boldsymbol\beta)=\underbrace{\|\mathbf{y}-X\boldsymbol\beta\|_2^2}_{\text{fit the data}}+\underbrace{\lambda\|\boldsymbol\beta\|_2^2}_{\text{discourage large coefficients}},\qquad\lambda\ge0.$$
:::

Since $\|\boldsymbol\beta\|^2=\boldsymbol\beta^T\boldsymbol\beta$, its derivative is

$$\frac{\partial}{\partial\boldsymbol\beta}(\lambda\boldsymbol\beta^T\boldsymbol\beta)=2\lambda\boldsymbol\beta.$$

Therefore

$$\nabla_{\boldsymbol\beta}L=-2X^T(\mathbf{y}-X\boldsymbol\beta)+2\lambda\boldsymbol\beta.$$

Set equal to zero:

$$\begin{aligned}-2X^T(\mathbf{y}-X\boldsymbol\beta)+2\lambda\boldsymbol\beta&=\mathbf{0}\\-X^T\mathbf{y}+X^TX\boldsymbol\beta+\lambda\boldsymbol\beta&=\mathbf{0}\\X^TX\boldsymbol\beta+\lambda\boldsymbol\beta&=X^T\mathbf{y}.\end{aligned}$$

To combine the two terms multiplying $\boldsymbol\beta$, write $\lambda\boldsymbol\beta=\lambda I\boldsymbol\beta$. Then

$$(X^TX+\lambda I)\boldsymbol\beta=X^T\mathbf{y}$$

and hence

$$\boxed{\hat{\boldsymbol\beta}_{\text{ridge}}=(X^TX+\lambda I)^{-1}X^T\mathbf{y}}$$

### Why does adding $\lambda I$ help?

Adding $\lambda I$ increases the diagonal entries of $X^TX$. In the lecture framing, this makes the matrix invertible even when predictors are collinear and improves numerical stability. For example,

$$X^TX=\begin{bmatrix}4&3.99\\3.99&4\end{bmatrix}\qquad\longrightarrow\qquad X^TX+I=\begin{bmatrix}5&3.99\\3.99&5\end{bmatrix}.$$

In eigenvalue language (§1.1.5): $X^TX+\lambda I$ has eigenvalues $\lambda_i+\lambda$, so every near-zero eigenvalue that made the inverse explode gets lifted safely away from 0.

:::ml Ridge tradeoff
Ridge deliberately introduces some **bias** in exchange for lower **variance** and more stable coefficients (the same trade-off as §2.3.2). It may accept a slightly worse fit to the training data in order to generalize more reliably.
:::

## 3.2.9 Ridge Worked Example from the Lecture

The lecture uses

$$X=\begin{bmatrix}1&1&1\\1&1&0.9\\1&0.9&1\\1&0.9&0.9\end{bmatrix},\qquad\mathbf{y}=\begin{bmatrix}2.8\\2.5\\2.6\\2.4\end{bmatrix}.$$

The non-intercept predictor columns are extremely similar, so $X^TX$ is nearly singular:

$$X^TX=\begin{bmatrix}4&3.8&3.8\\3.8&3.62&3.61\\3.8&3.61&3.62\end{bmatrix},\qquad X^T\mathbf{y}=\begin{bmatrix}10.3\\9.8\\9.81\end{bmatrix}.$$

(Its determinant is 0.0004 — practically the collapsed-dimension case from §1.1.1.) The lecture reports the OLS estimate

$$\hat{\boldsymbol\beta}_{\text{OLS}}\approx\begin{bmatrix}-1.225\\1.500\\2.500\end{bmatrix}.$$

The coefficients are comparatively large and uneven despite the predictor columns carrying nearly the same information.

Now set $\lambda=1$:

$$X^TX+I=\begin{bmatrix}5&3.8&3.8\\3.8&4.62&3.61\\3.8&3.61&4.62\end{bmatrix}\qquad\Rightarrow\qquad\hat{\boldsymbol\beta}_{\text{ridge}}\approx\begin{bmatrix}0.835\\0.801\\0.811\end{bmatrix}.$$

The coefficients are smaller and much more similar, reflecting the fact that the columns contain similar information.

:::fig coef-bars
:::

### What happens as $\lambda$ changes?

$$\boxed{\lambda\to0\;\Rightarrow\;\text{Ridge}\to\text{OLS}}\qquad\qquad\boxed{\lambda\to\infty\;\Rightarrow\;\hat{\boldsymbol\beta}_{\text{ridge}}\to\mathbf{0}}$$

A moderate $\lambda$ balances fit and stability. This is why $L_2$ regularization keeps weights *small but not zero* (§1.1.7): the circle has no corners, so Ridge shrinks rather than deletes.

:::viz ridge | Move $\lambda$ along the coefficient paths of the lecture example ($\lambda=0$ is OLS, $\lambda=1$ the lecture's ridge). "Nudge $\mathbf{y}$" adds tiny noise to the targets: OLS coefficients swing, ridge barely moves.
:::

## 3.2.10 Quick Reference: OLS vs. WLS vs. Ridge

|{stack}
| | OLS | WLS | Ridge |
|---|---|---|---|
| **Problem addressed** | baseline linear regression | unequal residual variance / reliability | multicollinearity and unstable coefficients |
| **Objective** | $\|\mathbf{y}-X\boldsymbol\beta\|^2$ | $(\mathbf{y}-X\boldsymbol\beta)^TW(\mathbf{y}-X\boldsymbol\beta)$ | $\|\mathbf{y}-X\boldsymbol\beta\|^2+\lambda\|\boldsymbol\beta\|^2$ |
| **Normal equations** | $X^TX\boldsymbol\beta=X^T\mathbf{y}$ | $X^TWX\boldsymbol\beta=X^TW\mathbf{y}$ | $(X^TX+\lambda I)\boldsymbol\beta=X^T\mathbf{y}$ |
| **Closed form** | $(X^TX)^{-1}X^T\mathbf{y}$ | $(X^TWX)^{-1}X^TW\mathbf{y}$ | $(X^TX+\lambda I)^{-1}X^T\mathbf{y}$ |
| **Key idea** | orthogonal projection onto $\operatorname{Col}(X)$ | weight observations by reliability | penalize large coefficients |

:::def One chain to remember
$$\text{OLS: closest point in }\operatorname{Col}(X)\;\Rightarrow\;\mathbf{e}\perp\operatorname{Col}(X)\;\Rightarrow\;X^T\mathbf{e}=\mathbf{0}\;\Rightarrow\;X^TX\boldsymbol\beta=X^T\mathbf{y}.$$

$$\begin{gathered}\text{WLS: unequal uncertainty}\Rightarrow W\text{ changes each residual's influence.}\\\text{Ridge: correlated predictors}\Rightarrow+\lambda I\text{ stabilizes the solution.}\end{gathered}$$
:::
