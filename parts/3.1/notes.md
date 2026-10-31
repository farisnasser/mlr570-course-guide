:::plain
**The flow:** perceptron limitation → maximum margin → line/hyperplane geometry → point-to-hyperplane distance → margin $2/\|\mathbf{w}\|$ → hard-margin primal → Lagrangian → dual problem → support vectors → worked example.
:::

:::add Added note
**Scope for this lecture.** These notes cover the hard-margin SVM material used in class. Soft-margin SVM and the hyperparameter $C$, as well as KKT conditions, are intentionally omitted.
:::

## 3.1.1 Why SVM? From "a separator" to the best separator

The perceptron can find a separating hyperplane when the data are linearly separable, but it may converge to **any** valid separator. It does not ask whether the boundary passes dangerously close to training points. SVM adds one central idea:

:::def Central idea
Among all separating hyperplanes, choose the one with the **largest geometric margin**.
:::

:::fig two-separators
:::

:::add Mental model
**Perceptron:** "Did I separate the classes?"\
**SVM:** "Can I separate them while leaving the **largest possible safety corridor** between the classes?"
:::

## 3.1.2 Rebuilding the Line Geometry from First Principles

Before the SVM formulas, we need to understand exactly what

$$\boxed{\mathbf{w}^T\mathbf{x}+b=0}$$

means geometrically.

### Start with an A-level line: $x+y=4$

Points such as $(1,3)$ and $(3,1)$ lie on the line because

$$1+3=4,\qquad3+1=4.$$

Subtracting the two points gives the displacement from one point to the other:

$$\underbrace{(3,1)}_{P_2}-\underbrace{(1,3)}_{P_1}=(2,-2).$$

This vector points **along** the line: move 2 units right and 2 units down.

:::fig line-normal
:::

The coefficients of the line are $(1,1)$. Take their dot product with the direction along the line:

$$(1,1)\cdot(2,-2)=2-2=0.$$

A zero dot product means perpendicular vectors, so $(1,1)$ is **normal** (perpendicular) to the line.

### Why the coefficients of $Ax+By+C=0$ form the normal vector

Take two arbitrary points $P_1=(x_1,y_1)$ and $P_2=(x_2,y_2)$ on

$$Ax+By+C=0.$$

They both satisfy the line equation:

$$Ax_1+By_1+C=0,\qquad Ax_2+By_2+C=0.$$

Subtract the equations:

$$A(x_2-x_1)+B(y_2-y_1)=0.$$

But $(x_2-x_1,\ y_2-y_1)$ is exactly the movement vector along the line. Therefore

$$(A,B)\cdot(x_2-x_1,\ y_2-y_1)=0.$$

Hence

$$\boxed{(A,B)\perp\text{the line}}$$

ML simply writes the same idea as

$$\mathbf{w}^T\mathbf{x}+b=0,$$

where $\mathbf{w}$ contains the line/hyperplane coefficients. Therefore

:::def Normal vector
$\mathbf{w}$ is **perpendicular** to the decision boundary.
:::

### Why $\mathbf{w}^T\mathbf{x}+b$ tells us which side a point is on

For any point $\mathbf{x}$, compute the scalar

$$s(\mathbf{x})=\mathbf{w}^T\mathbf{x}+b.$$

The boundary itself contains all points with $s(\mathbf{x})=0$. Moving in the direction of $\mathbf{w}$ **increases** the score because

$$s(\mathbf{x}+\epsilon\mathbf{w})=\mathbf{w}^T(\mathbf{x}+\epsilon\mathbf{w})+b=s(\mathbf{x})+\epsilon\|\mathbf{w}\|^2>s(\mathbf{x})$$

for $\epsilon>0$. So $\mathbf{w}$ points toward the positive side.

:::def Signed side-test
$$\begin{aligned}\mathbf{w}^T\mathbf{x}+b>0&\;\Rightarrow\;\text{positive side}\\\mathbf{w}^T\mathbf{x}+b<0&\;\Rightarrow\;\text{negative side}\\\mathbf{w}^T\mathbf{x}+b=0&\;\Rightarrow\;\text{on the boundary}\end{aligned}$$

Think of $\mathbf{w}^T\mathbf{x}+b$ as a **signed side-test** along the normal direction.
:::

E.g. for $x+y-4=0$: the point $(3,3)$ gives $3+3-4=2>0$ (the side $\mathbf{n}$ points to), and $(0,0)$ gives $-4<0$.

## 3.1.3 Distance from a Point to a Hyperplane

For a point $\mathbf{x}_0$ and the hyperplane $\mathbf{w}^T\mathbf{x}+b=0$, the perpendicular distance is

:::def Point-to-hyperplane distance
$$\operatorname{dist}(\mathbf{x}_0,\ \mathbf{w}^T\mathbf{x}+b=0)=\frac{|\mathbf{w}^T\mathbf{x}_0+b|}{\|\mathbf{w}\|}$$
:::

:::ml Connection to the familiar 2D formula
For $Ax+By+C=0$ and point $(x_0,y_0)$,

$$\frac{|Ax_0+By_0+C|}{\sqrt{A^2+B^2}}$$

is exactly the same formula. In ML notation $(A,B)$ becomes $\mathbf{w}$, and $\sqrt{A^2+B^2}$ is the Pythagorean length $\|\mathbf{w}\|$ (the $L_2$ norm, §1.1.7).
:::

:::ex Example
Distance from $(0,0)$ to $x+y-4=0$:

$$\frac{|0+0-4|}{\sqrt{1^2+1^2}}=\frac{4}{\sqrt2}=2\sqrt2\approx2.83.$$
:::

### Why the denominator is necessary

The raw value $\mathbf{w}^T\mathbf{x}_0+b$ is **not** a true geometric distance because the same line can be written with different scales. For example,

$$x+y-1=0\qquad\text{and}\qquad10x+10y-10=0$$

are the same line, but the raw score of every point is multiplied by 10 in the second equation. The norm $\|\mathbf{w}\|$ is also multiplied by 10, so the ratio stays unchanged.

:::add Key idea
The numerator measures separation from the boundary in the normal direction, but it is scaled by the size of $\mathbf{w}$. Dividing by $\|\mathbf{w}\|$ removes that arbitrary scaling and gives **actual geometric distance**.
:::

### Short derivation using the normal direction

:::cols
:::col
Let $\mathbf{z}$ be the closest point on the hyperplane to $\mathbf{x}_0$. The shortest route is perpendicular to the hyperplane, so it must be parallel to $\mathbf{w}$:

$$\mathbf{x}_0-\mathbf{z}=\lambda\mathbf{w}.$$

Because $\mathbf{z}$ lies on the hyperplane,

$$\mathbf{w}^T\mathbf{z}+b=0.$$
:::
:::col
:::fig foot
:::
:::
:::

Now

$$\mathbf{w}^T\mathbf{x}_0+b=\mathbf{w}^T(\mathbf{z}+\lambda\mathbf{w})+b=\underbrace{(\mathbf{w}^T\mathbf{z}+b)}_{0}+\lambda\mathbf{w}^T\mathbf{w}=\lambda\|\mathbf{w}\|^2.$$

Thus

$$|\lambda|=\frac{|\mathbf{w}^T\mathbf{x}_0+b|}{\|\mathbf{w}\|^2}.$$

The geometric distance is

$$\|\mathbf{x}_0-\mathbf{z}\|=\|\lambda\mathbf{w}\|=|\lambda|\,\|\mathbf{w}\|=\boxed{\frac{|\mathbf{w}^T\mathbf{x}_0+b|}{\|\mathbf{w}\|}}$$

:::viz line | Drag the tip of $\mathbf{w}$, the point $\mathbf{x}_0$, and $b$. The colour is the signed score $s(\mathbf{x})$; scaling $(\mathbf{w},b)$ multiplies the raw score but leaves the distance alone.
:::

## 3.1.4 SVM Margin Geometry

SVM chooses a convenient scaling in which the two **margin hyperplanes** are

$$\boxed{\mathbf{w}^T\mathbf{x}+b=+1},\qquad\boxed{\mathbf{w}^T\mathbf{x}+b=-1},$$

with the **decision boundary** in the middle:

$$\boxed{\mathbf{w}^T\mathbf{x}+b=0}$$

:::fig margin
:::

A point on $\mathbf{w}^T\mathbf{x}+b=+1$ has distance

$$\frac{|1|}{\|\mathbf{w}\|}=\frac{1}{\|\mathbf{w}\|}$$

from the center boundary. Likewise, the $-1$ margin is $1/\|\mathbf{w}\|$ away. Therefore

:::def Margin width
$$\text{total margin}=\frac{2}{\|\mathbf{w}\|}$$
:::

### Why specifically $+1$ and $-1$?

The classifier only depends on the **sign** of $\mathbf{w}^T\mathbf{x}+b$. Multiplying $\mathbf{w}$ and $b$ by any positive constant does not change the geometric boundary (same as $x+y-1=0$ vs $10x+10y-10=0$). Therefore the scale of $\mathbf{w},b$ is arbitrary until we fix it. SVM chooses the convenient normalization in which the **closest** positive and negative points have scores $+1$ and $-1$.

:::add The number 1 is not magical
We could use another positive constant, but every parameter would simply be rescaled. Choosing $\pm1$ makes the margin formulas and constraints especially clean.
:::

### From maximum margin to the SVM objective

We want to maximize

$$\frac{2}{\|\mathbf{w}\|}.$$

A fraction becomes larger when its positive denominator becomes smaller, so

$$\boxed{\max\frac{2}{\|\mathbf{w}\|}\iff\min\|\mathbf{w}\|}$$

Instead of minimizing $\|\mathbf{w}\|$ directly, SVM uses

$$\boxed{\min_{\mathbf{w},b}\ \frac12\|\mathbf{w}\|^2}$$

because it has the same minimizer and a simpler derivative (§1.3.5):

$$\frac{\partial}{\partial\mathbf{w}}\left(\frac12\mathbf{w}^T\mathbf{w}\right)=\mathbf{w}.$$

($\|\mathbf{w}\|$ itself has a square root, which is messier to differentiate and not smooth at $\mathbf{w}=\mathbf{0}$. Squaring is monotone for nonnegative numbers, so the minimizer doesn't move.)

## 3.1.5 Why We Need the Hard-Margin Constraint

If the objective were only

$$\min\ \frac12\|\mathbf{w}\|^2,$$

the smallest vector is trivially $\mathbf{w}=\mathbf{0}$. But then

$$\mathbf{w}^T\mathbf{x}+b=b,$$

so the prediction no longer depends on $\mathbf{x}$ at all. There is no meaningful oriented decision boundary.

SVM therefore says: make $\|\mathbf{w}\|$ as small as possible *while still correctly separating every training point outside the margin.*

For a positive point ($y_i=+1$), require

$$\mathbf{w}^T\mathbf{x}_i+b\ge1.$$

For a negative point ($y_i=-1$), require

$$\mathbf{w}^T\mathbf{x}_i+b\le-1.$$

Both cases combine into one compact inequality:

:::def Hard-margin constraint
$$y_i(\mathbf{w}^T\mathbf{x}_i+b)\ge1$$
:::

:::ex Check the two cases
$$\begin{aligned}y_i=+1:&\quad(+1)(\mathbf{w}^T\mathbf{x}_i+b)\ge1\;\Rightarrow\;\mathbf{w}^T\mathbf{x}_i+b\ge1.\\y_i=-1:&\quad(-1)(\mathbf{w}^T\mathbf{x}_i+b)\ge1\;\Rightarrow\;\mathbf{w}^T\mathbf{x}_i+b\le-1.\end{aligned}$$

The same expression therefore places both classes on the correct side of their corresponding margin boundary.
:::

:::add Why $\mathbf{w}=\mathbf{0}$ cannot survive the constraints
If $\mathbf{w}=\mathbf{0}$, then every score is just $b$. A positive sample would require $b\ge1$, while a negative sample would require $b\le-1$. With both classes present, both conditions cannot hold at once.
:::

## 3.1.6 Hard-Margin SVM Primal Problem

Putting the objective and constraints together gives the **primal** optimization problem:

:::def Primal
$$\begin{aligned}\min_{\mathbf{w},b}\quad&\frac12\|\mathbf{w}\|^2\\\text{subject to}\quad&y_i(\mathbf{w}^T\mathbf{x}_i+b)\ge1,\qquad i=1,\dots,n.\end{aligned}$$
:::

:::ml Interpretation
The **objective** wants the widest possible margin. The **constraints** prevent the optimizer from achieving that by destroying the classifier; every training point must remain correctly separated and outside the margin.
:::

This is a convex problem (§1.3.6): a quadratic bowl in $\mathbf{w}$ with linear constraints, so there is one global optimum and no bad local minima.

## 3.1.7 Lagrange Multipliers: Putting the Constraints into One Function

The primal is constrained. A **Lagrange multiplier** gives each constraint a nonnegative weight $\alpha_i\ge0$ and combines objective and constraints into a single **Lagrangian**.

Rewrite each constraint as

$$g_i(\mathbf{w},b)=y_i(\mathbf{w}^T\mathbf{x}_i+b)-1\ge0.$$

Then

:::def Lagrangian
$$\mathcal{L}(\mathbf{w},b,\boldsymbol\alpha)=\frac12\|\mathbf{w}\|^2-\sum_{i=1}^n\alpha_i\Big[y_i(\mathbf{w}^T\mathbf{x}_i+b)-1\Big]\qquad\text{with }\alpha_i\ge0.$$
:::

### Expand the Lagrangian slowly

First distribute $y_i$ inside the brackets:

$$y_i(\mathbf{w}^T\mathbf{x}_i+b)-1=y_i\mathbf{w}^T\mathbf{x}_i+y_ib-1.$$

Therefore

$$\mathcal{L}=\frac12\|\mathbf{w}\|^2-\sum_i\alpha_i\big(y_i\mathbf{w}^T\mathbf{x}_i+y_ib-1\big).$$

Now distribute the minus sign:

$$\boxed{\mathcal{L}=\frac12\|\mathbf{w}\|^2-\sum_i\alpha_iy_i\mathbf{w}^T\mathbf{x}_i-\sum_i\alpha_iy_ib+\sum_i\alpha_i}$$

No new SVM idea happened here; this is simply the algebraic identity $-(A+B-C)=-A-B+C$ applied inside the sum.

### Differentiate with respect to $\mathbf{w}$ and $b$

For $\mathbf{w}$ (using $\nabla_{\mathbf{w}}\frac12\mathbf{w}^T\mathbf{w}=\mathbf{w}$ and $\nabla_{\mathbf{w}}\mathbf{w}^T\mathbf{x}_i=\mathbf{x}_i$, §1.3.5),

$$\frac{\partial\mathcal{L}}{\partial\mathbf{w}}=\mathbf{w}-\sum_i\alpha_iy_i\mathbf{x}_i.$$

Set it to zero:

$$\boxed{\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i}$$

For $b$,

$$\frac{\partial\mathcal{L}}{\partial b}=-\sum_i\alpha_iy_i.$$

Set it to zero:

$$\boxed{\sum_i\alpha_iy_i=0}$$

:::ml Important meaning
The final normal vector $\mathbf{w}$ is a **weighted combination of training samples**. This is the first hint that we may be able to solve the problem using relationships between training samples rather than constructing $\mathbf{w}$ directly.
:::

## 3.1.8 From the Lagrangian to the Dual Problem

The **primal** solves directly for $\mathbf{w},b$. The **dual** rewrites the same optimization problem so that we solve for the constraint weights $\alpha_1,\dots,\alpha_n$ instead.

### Where do the indices $i$ and $j$ come from?

We have

$$\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i.$$

To compute $\|\mathbf{w}\|^2=\mathbf{w}^T\mathbf{w}$, we multiply one copy of this sum by another copy:

$$\Big(\sum_i\alpha_iy_i\mathbf{x}_i\Big)^T\Big(\sum_j\alpha_jy_j\mathbf{x}_j\Big).$$

We use $i$ for the sample chosen from the first sum and $j$ for the sample chosen from the second sum. With two samples, the double sum simply represents the four pairings $(1,1),(1,2),(2,1),(2,2)$.

### Why is only $\mathbf{x}_i$ visibly transposed?

$\alpha_i$ and $y_i$ are scalars, so transposing them changes nothing. $\mathbf{x}_i$ is a column vector, so its transpose is a row vector. Therefore

$$(\alpha_iy_i\mathbf{x}_i)^T=\alpha_iy_i\mathbf{x}_i^T.$$

Hence

$$\boxed{\|\mathbf{w}\|^2=\sum_i\sum_j\alpha_i\alpha_jy_iy_j\,\mathbf{x}_i^T\mathbf{x}_j}$$

### Substitute $\mathbf{w}$ back into the Lagrangian

From the expanded Lagrangian,

$$\mathcal{L}=\frac12\|\mathbf{w}\|^2-\sum_i\alpha_iy_i\mathbf{w}^T\mathbf{x}_i-b\sum_i\alpha_iy_i+\sum_i\alpha_i.$$

The $b$ term vanishes because

$$\sum_i\alpha_iy_i=0.$$

For the middle term, substitute $\mathbf{w}=\sum_j\alpha_jy_j\mathbf{x}_j$:

$$\sum_i\alpha_iy_i\mathbf{w}^T\mathbf{x}_i=\sum_i\sum_j\alpha_i\alpha_jy_iy_j\,\mathbf{x}_j^T\mathbf{x}_i=\|\mathbf{w}\|^2.$$

So the Lagrangian becomes $\frac12\|\mathbf{w}\|^2-\|\mathbf{w}\|^2+\sum_i\alpha_i=\sum_i\alpha_i-\frac12\|\mathbf{w}\|^2$. Collecting the matching double sums, we obtain the **dual function**

:::def Dual function
$$g(\boldsymbol\alpha)=\sum_i\alpha_i-\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_j\,\mathbf{x}_i^T\mathbf{x}_j$$
:::

## 3.1.9 Why Does the Dual Maximize?

This is the conceptual step that makes the dual formulation make sense.

The original primal **minimizes** an objective $f(\mathbf{w},b)$ subject to constraints $g_i(\mathbf{w},b)\ge0$. Its Lagrangian is

$$\mathcal{L}(\mathbf{w},b,\boldsymbol\alpha)=f(\mathbf{w},b)-\sum_i\alpha_ig_i(\mathbf{w},b),\qquad\alpha_i\ge0.$$

For any **feasible** point, $g_i(\mathbf{w},b)\ge0$. Since $\alpha_i\ge0$,

$$-\alpha_ig_i(\mathbf{w},b)\le0.$$

Therefore

$$\boxed{\mathcal{L}(\mathbf{w},b,\boldsymbol\alpha)\le f(\mathbf{w},b)}$$

for feasible solutions. Now define

$$\boxed{g(\boldsymbol\alpha)=\min_{\mathbf{w},b}\mathcal{L}(\mathbf{w},b,\boldsymbol\alpha)}$$

Because we minimize the Lagrangian over $\mathbf{w},b$, $g(\boldsymbol\alpha)$ gives a **lower bound** on the primal optimum $p^*$:

$$\boxed{g(\boldsymbol\alpha)\le p^*}$$

Different choices of $\boldsymbol\alpha$ give different lower bounds. We want the one that comes as close as possible to the true primal optimum, so we choose the **largest** lower bound:

$$\boxed{\max_{\boldsymbol\alpha}\ g(\boldsymbol\alpha)}$$

:::fig bounds
:::

:::ex Toy optimization to see the max appear
:::cols
:::col
Consider

$$\min_x\ x^2\quad\text{subject to}\quad x\ge1.$$

The primal optimum is obviously $x=1$ with $p^*=1$.\
Write the constraint as $x-1\ge0$ and form

$$\mathcal{L}(x,\alpha)=x^2-\alpha(x-1),\qquad\alpha\ge0.$$

For fixed $\alpha$, minimizing over $x$ gives $2x-\alpha=0$, so $x=\alpha/2$. Substitution gives
:::
:::col
:::fig toy-dual
:::
:::
:::

$$g(\alpha)=\frac{\alpha^2}{4}-\alpha\Big(\frac\alpha2-1\Big)=\alpha-\frac{\alpha^2}{4}.$$

Then

$$g(0)=0,\qquad g(1)=0.75,\qquad g(2)=1,\qquad g(3)=0.75.$$

Every value is a lower bound on $p^*=1$, and the best one is obtained by **maximizing** $g(\alpha)$. Here the max, $g(2)=1$, hits $p^*$ exactly: for convex problems like this one (and the SVM), the best lower bound is not just close, it equals the primal optimum.
:::

:::viz dual | Slide $\alpha$. Left: $\mathcal{L}(x,\alpha)$ as a function of $x$; its lowest value is $g(\alpha)$. Right: $g(\alpha)$ itself, always at or below $p^*=1$, touching it at $\alpha=2$.
:::

:::add Memory line
The primal **minimizes** the original objective. The dual **maximizes** the tightest lower bound on that minimum.
:::

## 3.1.10 Final Hard-Margin Dual Problem

:::def SVM dual
$$\begin{aligned}\max_{\boldsymbol\alpha}\quad&\sum_{i=1}^n\alpha_i-\frac12\sum_{i=1}^n\sum_{j=1}^n\alpha_i\alpha_jy_iy_j\,\mathbf{x}_i^T\mathbf{x}_j\\\text{subject to}\quad&\sum_{i=1}^n\alpha_iy_i=0,\\&\alpha_i\ge0,\qquad i=1,\dots,n.\end{aligned}$$
:::

:::ml Why the dual matters later
The training data appear **only through dot products** $\mathbf{x}_i^T\mathbf{x}_j$. This is the doorway to **kernel methods**: later, $\mathbf{x}_i^T\mathbf{x}_j$ can be replaced by a kernel $K(\mathbf{x}_i,\mathbf{x}_j)=\phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)$ without explicitly constructing the high-dimensional feature map.
:::

Once the $\alpha_i$ are found, $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$, so a new point is classified by

$$\operatorname{sign}\big(\mathbf{w}^T\mathbf{x}+b\big)=\operatorname{sign}\Big(\sum_i\alpha_iy_i\,\mathbf{x}_i^T\mathbf{x}+b\Big),$$

which again only needs dot products between the new point and training points.

## 3.1.11 Support Vectors and the Geometry of the Final Classifier

:::def Support vectors
The training points that **touch the margin hyperplanes**. Geometrically, they satisfy

$$y_i(\mathbf{w}^T\mathbf{x}_i+b)=1.$$

They are the closest points to the decision boundary and therefore limit how wide the margin can become.
:::

:::fig support-vectors
:::

Points far from the boundary are **not** what limits the margin. Moving a non-support-vector point slightly usually leaves the maximum-margin separator unchanged. In the dual representation, the support vectors are the training samples that retain **nonzero coefficients** ($\alpha_i>0$); every other point has $\alpha_i=0$ and drops out of $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$. We use this result geometrically here without the KKT derivation.

:::add Mental model
SVM largely **ignores the "easy" points** far away from the boundary. The difficult, closest points determine the separator.
:::

## 3.1.12 Lecture Worked Example

The lecture uses ten linearly separable points:

$$\begin{aligned}D_+&=\{(2,2),\ (3,1.6),\ (2.6,2.6),\ (3,2.6),\ (2,3)\},\\D_-&=\{(-1,-1),\ (0,-2),\ (-2,0),\ (-1.4,-1.2),\ (0.6,-1.6)\}.\end{aligned}$$

One valid separator is $x_1+x_2=0.5$, but SVM seeks the **maximum-margin** separator rather than merely any separator.

The exact hard-margin solution given in the lecture is

$$\boxed{\mathbf{w}^*=\Big(\frac{8}{29},\ \frac{13}{29}\Big)\approx(0.276,\ 0.448),\qquad b^*=-\frac{13}{29}\approx-0.448}$$

Thus the decision boundary is (multiplying through by 29)

$$\boxed{8x_1+13x_2-13=0}$$

The geometric margin is

$$\|\mathbf{w}^*\|=\frac{\sqrt{8^2+13^2}}{29}=\frac{\sqrt{233}}{29}\approx0.526,\qquad\boxed{\frac{2}{\|\mathbf{w}^*\|}\approx3.80}$$

:::fig worked
:::

Exactly three points lie on the margins and are the **support vectors** (circled):

$$\boxed{(2,2),\qquad(-2,0),\qquad(0.6,-1.6)}$$

They satisfy $y_i(\mathbf{w}^{*T}\mathbf{x}_i+b^*)=1$ and determine the geometry of the final separator.

:::ex Check: $\mathbf{w}^{*T}\mathbf{x}+b^*=(8x_1+13x_2-13)/29$
| Point | $y$ | $8x_1+13x_2-13$ | $y(\mathbf{w}^{*T}\mathbf{x}+b^*)$ |
|:-:|:-:|:-:|:-:|
| $(2,2)$ | $+1$ | $16+26-13=29$ | **1** (support vector) |
| $(-2,0)$ | $-1$ | $-16+0-13=-29$ | **1** (support vector) |
| $(0.6,-1.6)$ | $-1$ | $4.8-20.8-13=-29$ | **1** (support vector) |
| $(3,1.6)$ | $+1$ | $24+20.8-13=31.8$ | 1.10 |
| $(-1,-1)$ | $-1$ | $-8-13-13=-34$ | 1.17 |

Support vectors give exactly 1; every other point gives more than 1 (it sits beyond its margin). The remaining points $(2.6,2.6)$, $(3,2.6)$, $(2,3)$, $(0,-2)$, $(-1.4,-1.2)$ also give values $>1$.
:::

:::viz svm | The ten lecture points with the exact maximum-margin separator, recomputed whenever you drag a point. Move a point far from the boundary and nothing changes; move a support vector and the whole corridor moves.
:::

## 3.1.13 Quick Reference

|{stack}
| **Concept** | What to remember |
|---|---|
| **Decision boundary** | $\mathbf{w}^T\mathbf{x}+b=0$. The vector $\mathbf{w}$ is perpendicular to the hyperplane and points toward the positive side. |
| **Signed side-test** | $\mathbf{w}^T\mathbf{x}+b>0$ positive side; $<0$ negative side; $=0$ on the boundary. |
| **Point-to-hyperplane distance** | $\lvert\mathbf{w}^T\mathbf{x}+b\rvert/\|\mathbf{w}\|$. The denominator removes arbitrary scaling of the line equation. |
| **Margin planes** | $\mathbf{w}^T\mathbf{x}+b=+1$ and $\mathbf{w}^T\mathbf{x}+b=-1$. The choice $\pm1$ is a convenient normalization. |
| **Margin width** | $2/\|\mathbf{w}\|$. Therefore maximum margin is equivalent to minimizing $\|\mathbf{w}\|$. |
| **Primal objective** | $\min\frac12\|\mathbf{w}\|^2$. The $\frac12$ and square simplify differentiation without changing the minimizer. |
| **Hard-margin constraint** | $y_i(\mathbf{w}^T\mathbf{x}_i+b)\ge1$. Positive points stay beyond $+1$ and negative points beyond $-1$. |
| **Lagrangian** | $\mathcal{L}=\frac12\|\mathbf{w}\|^2-\sum_i\alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i+b)-1]$, $\ \alpha_i\ge0$. |
| **Stationarity results** | $\mathbf{w}=\sum_i\alpha_iy_i\mathbf{x}_i$ and $\sum_i\alpha_iy_i=0$. |
| **Dual** | $\max_{\boldsymbol\alpha}\sum_i\alpha_i-\frac12\sum_i\sum_j\alpha_i\alpha_jy_iy_j\mathbf{x}_i^T\mathbf{x}_j$, s.t. $\sum_i\alpha_iy_i=0$, $\alpha_i\ge0$. |
| **Dual idea** | Solve for $\alpha_i$ instead of $\mathbf{w},b$; maximize the tightest lower bound on the primal minimum. |
| **Kernel bridge** | The dual uses samples only through dot products $\mathbf{x}_i^T\mathbf{x}_j$. |
| **Support vectors** | Closest points on the margins ($\alpha_i>0$); they determine the maximum-margin separator. |

:::def One-line summary
SVM chooses the separating hyperplane whose closest training points are as far from the boundary as possible, and the dual reformulates this geometry in terms of pairwise dot products between training samples.
:::
