:::plain
**The flow:** Limit → derivative → partial derivative → gradient → Jacobian → vector/matrix derivatives → chain rule → matrix identities → convexity → optimization.
:::

The goal of this part is not to relearn an entire calculus course. We only need the pieces that explain how a machine-learning loss changes when we change its parameters. In ML, derivatives are fundamentally **sensitivity measurements**.

:::ml Why calculus matters in ML
A model has parameters $\boldsymbol\theta$ and a loss $L(\boldsymbol\theta)$. Training asks: "If I change each parameter a tiny amount, which way makes the loss go down?" The derivative/gradient answers exactly that question.
:::

## 1.3.1 Limits & Derivatives

:::add Added note
Limits are listed in the course outline but are not developed in the Parr–Howard matrix-calculus paper. Only the amount needed to understand the derivative definition is included here.
:::

### Limits

A limit describes what a function approaches as its input approaches some value:

$$\boxed{\lim_{x\to a}f(x)=L}$$

It asks what happens *near $a$*; the value $f(a)$ itself can even be missing. Limits let us define an instantaneous rate of change by shrinking a finite change down toward zero.

### Derivative = local rate of change

For $y=f(x)$, the derivative is

:::def Derivative
$$f'(x)=\frac{df}{dx}=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$$

It is the slope of the tangent line, or equivalently the **sensitivity of the output to a tiny change in the input**.
:::

:::fig secant-tangent
:::

For a small change $\Delta x$,

$$\boxed{f(x+\Delta x)\approx f(x)+f'(x)\,\Delta x}$$

So if $f'(3)=6$, then near $x=3$ a change $\Delta x=0.01$ changes the output by roughly $6(0.01)=0.06$.

:::ex Example
For $f(x)=x^2$,

$$\frac{df}{dx}=2x.$$

At $x=3$, the slope is 6. The derivative is not just a formula; it says how strongly the output reacts *at that point*.
:::

:::viz secant | Drag $x$ and shrink $h$: the secant slope $\frac{f(x+h)-f(x)}{h}$ closes in on the tangent slope $f'(x)=2x$.
:::

### Scalar derivative rules

|{stack}
| **Rule** | Formula | Example |
|---|---|---|
| **Constant** | $\frac{d}{dx}c=0$ | $\frac{d}{dx}99=0$ |
| **Constant multiple** | $\frac{d}{dx}[cf]=cf'$ | $\frac{d}{dx}(3x)=3$ |
| **Power** | $\frac{d}{dx}x^n=nx^{n-1}$ | $\frac{d}{dx}x^3=3x^2$ |
| **Sum/difference** | $(f\pm g)'=f'\pm g'$ | $\frac{d}{dx}(x^2+3x)=2x+3$ |
| **Product** | $(fg)'=f'g+fg'$ | $\frac{d}{dx}(x^2x)=3x^2$ |
| **Quotient** | $\big(\frac fg\big)'=\frac{f'g-fg'}{g^2}$ | $\frac{d}{dx}\frac1x=-\frac{1}{x^2}$ |
| **Chain** | $\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}$ | $y=\sin(x^2)\to2x\cos(x^2)$ |

:::ml ML interpretation
$\dfrac{\partial L}{\partial w_i}$ means: if weight $w_i$ moves a tiny amount, how strongly and in which direction does the loss $L$ move?
:::

## 1.3.2 Partial Derivatives & Gradients

Neural-network functions depend on many parameters, so one ordinary derivative is no longer enough.

### Partial derivatives

For $f(x,y)$, we differentiate with respect to **one variable at a time**, treating the others as constants:

$$\frac{\partial f}{\partial x}=\text{sensitivity to }x\text{ while }y\text{ is fixed},\qquad\frac{\partial f}{\partial y}=\text{sensitivity to }y\text{ while }x\text{ is fixed}.$$

:::ex Example: $f(x,y)=3x^2y$
With respect to $x$, treat $3y$ as constant:

$$\boxed{\frac{\partial f}{\partial x}=6xy}$$

With respect to $y$, treat $3x^2$ as constant:

$$\boxed{\frac{\partial f}{\partial y}=3x^2}$$
:::

### Gradient

For one scalar output $f(\mathbf{x})$ with many inputs, collect all partial derivatives into one vector.

:::def Gradient
$$\nabla f(\mathbf{x})=\begin{bmatrix}\dfrac{\partial f}{\partial x_1}&\dfrac{\partial f}{\partial x_2}&\cdots&\dfrac{\partial f}{\partial x_n}\end{bmatrix}$$

Following the Parr–Howard paper's **numerator-layout convention**, the gradient of a scalar output is written as a **row** vector. Some ML texts use a column gradient instead; that is simply the transpose. Be consistent with shapes.
:::

For $f(x,y)=3x^2y$,

$$\boxed{\nabla f=[\,6xy,\ 3x^2\,]}$$

:::ml Geometric meaning
The gradient points in the direction of **steepest increase**. Therefore $-\nabla f$ points in the direction of steepest local **decrease**. This is the central idea behind gradient descent.
:::

For a small vector change $\Delta\mathbf{x}$,

$$\boxed{\Delta f\approx\nabla f\,\Delta\mathbf{x}}$$

(in the paper's row-gradient convention). This is the multivariable version of $\Delta f\approx f'(x)\,\Delta x$.

:::viz partials | Drag to orbit; move the point $(x,y)$. The red cut holds $y$ fixed (slope $\partial f/\partial x$), the blue cut holds $x$ fixed (slope $\partial f/\partial y$); on the contour map $\nabla f$ crosses the contour at right angles.
:::

## 1.3.3 Vector & Matrix Derivatives

### From derivative $\to$ gradient $\to$ Jacobian

| Mapping | Derivative object | Shape |
|---|---|---|
| scalar $\to$ scalar | ordinary derivative | $1\times1$ |
| vector $\to$ scalar | gradient | $1\times n$ |
| vector $\to$ vector | Jacobian | $m\times n$ |

Suppose $\mathbf{x}\in\mathbb{R}^n$ and $\mathbf{y}=\mathbf{f}(\mathbf{x})\in\mathbb{R}^m$, where each output $y_i=f_i(\mathbf{x})$ is a scalar function.

:::def Jacobian
$$J=\frac{\partial\mathbf{y}}{\partial\mathbf{x}}=\begin{bmatrix}\dfrac{\partial y_1}{\partial x_1}&\cdots&\dfrac{\partial y_1}{\partial x_n}\\\vdots&\ddots&\vdots\\\dfrac{\partial y_m}{\partial x_1}&\cdots&\dfrac{\partial y_m}{\partial x_n}\end{bmatrix}$$

Each **row** is the gradient of one output. Therefore

$$J\in\mathbb{R}^{m\times n}:\qquad\text{one row per output, one column per input.}$$
:::

:::ex Example
Let

$$f(x,y)=3x^2y,\qquad g(x,y)=2x+y^8.$$

Then

$$\nabla f=[\,6xy,\ 3x^2\,],\qquad\nabla g=[\,2,\ 8y^7\,],$$

so

$$\boxed{J=\begin{bmatrix}6xy&3x^2\\2&8y^7\end{bmatrix}}$$
:::

:::viz jacobian | Move the point and the size of the small square. The curved shape is the true image of the square under $(f,g)$; the parallelogram is its image under $J$. They agree more and more as the square shrinks.
:::

For the identity function $\mathbf{y}=\mathbf{x}$,

$$\frac{\partial x_i}{\partial x_j}=\begin{cases}1&i=j\\0&i\neq j\end{cases}\qquad\Rightarrow\qquad\boxed{\frac{\partial\mathbf{x}}{\partial\mathbf{x}}=I}$$

### Element-wise operations $\Rightarrow$ diagonal Jacobians

If each output $y_i$ depends only on the matching input $x_i$, then

$$\frac{\partial y_i}{\partial x_j}=0\qquad(i\neq j),$$

so all off-diagonal Jacobian entries vanish.

:::def Element-wise diagonal condition
$$y_i\text{ depends only on }x_i\quad\Rightarrow\quad J\text{ is diagonal.}$$
:::

Useful cases:

$$\boxed{\frac{\partial(\mathbf{w}+\mathbf{x})}{\partial\mathbf{w}}=I,\qquad\frac{\partial(\mathbf{w}+\mathbf{x})}{\partial\mathbf{x}}=I}$$

$$\boxed{\frac{\partial(\mathbf{w}\odot\mathbf{x})}{\partial\mathbf{w}}=\operatorname{diag}(\mathbf{x}),\qquad\frac{\partial(\mathbf{w}\odot\mathbf{x})}{\partial\mathbf{x}}=\operatorname{diag}(\mathbf{w})}$$

Here $\odot$ is element-wise (Hadamard) multiplication. This ties back to diagonal matrices (§1.1.4): each direction is handled on its own, nothing gets mixed.

### Scalar expansion

A scalar interacting with a vector is implicitly copied across all entries:

$$\mathbf{x}+z=\begin{bmatrix}x_1+z\\\vdots\\x_n+z\end{bmatrix},\qquad z\mathbf{x}=\begin{bmatrix}zx_1\\\vdots\\zx_n\end{bmatrix}.$$

Hence

$$\boxed{\frac{\partial(\mathbf{x}+z)}{\partial\mathbf{x}}=I,\qquad\frac{\partial(\mathbf{x}+z)}{\partial z}=\mathbf{1}}\qquad\text{and}\qquad\boxed{\frac{\partial(z\mathbf{x})}{\partial\mathbf{x}}=zI,\qquad\frac{\partial(z\mathbf{x})}{\partial z}=\mathbf{x}}$$

### Vector reduction: vector $\to$ scalar

Summing a vector gives a scalar:

$$y=\sum_ix_i.$$

Every $x_i$ contributes with derivative 1, so

$$\boxed{\frac{\partial}{\partial\mathbf{x}}\sum_ix_i=\mathbf{1}^T}$$

The dot product is a sum of element-wise products:

$$y=\mathbf{w}^T\mathbf{x}=\sum_iw_ix_i.$$

Therefore

$$\boxed{\frac{\partial(\mathbf{w}^T\mathbf{x})}{\partial\mathbf{x}}=\mathbf{w}^T,\qquad\frac{\partial(\mathbf{w}^T\mathbf{x})}{\partial\mathbf{w}}=\mathbf{x}^T}$$

under the paper's row-gradient convention.

:::ml Shape check
Never treat dimensions as decoration. If $\mathbf{y}\in\mathbb{R}^m$ and $\mathbf{x}\in\mathbb{R}^n$, then $\partial\mathbf{y}/\partial\mathbf{x}$ **must** be $m\times n$. Shape checking catches many matrix-calculus mistakes immediately.
:::

## 1.3.4 The Chain Rule

The chain rule handles **nested computations**. Deep networks are just long chains of nested transformations, so this is the mathematical core of backpropagation.

### 1) Single-variable chain rule

If

$$u=g(x),\qquad y=f(u),$$

then

:::def Chain rule
$$\frac{dy}{dx}=\frac{dy}{du}\,\frac{du}{dx}$$
:::

:::ex Example: $y=\sin(x^2)$
Let $u=x^2$, $y=\sin u$. Then

$$\frac{du}{dx}=2x,\qquad\frac{dy}{du}=\cos u,$$

so

$$\boxed{\frac{dy}{dx}=2x\cos(x^2)}$$
:::

The practical recipe is: **break into simple intermediate variables $\to$ differentiate each local step $\to$ multiply the local derivatives.**

### 2) Total derivative: when there are multiple paths

Sometimes $x$ affects the output in more than one way. Example:

$$y=x+x^2.$$

Let $u=x^2$, so $y=x+u$. There is a **direct** path $x\to y$ and an **indirect** path $x\to u\to y$.

:::fig total-derivative-graph
:::

:::def Total-derivative idea
Add the contribution from **every path** by which $x$ affects the output:

$$\frac{dy}{dx}=\frac{\partial y}{\partial x}+\frac{\partial y}{\partial u}\,\frac{du}{dx}$$

For $y=x+x^2$: $\ 1+(1)(2x)=1+2x$.
:::

More generally, if $f$ depends directly on $x$ and indirectly through $u_1,\dots,u_n$,

$$\boxed{\frac{df}{dx}=\frac{\partial f}{\partial x}+\sum_{i=1}^n\frac{\partial f}{\partial u_i}\,\frac{du_i}{dx}}$$

:::viz chain | Move $x$. Values flow forward along the arrows; local derivatives are multiplied backward along each path, and the paths are added. The last line checks the result against a tiny finite nudge.
:::

### 3) Vector chain rule

Let

$$\mathbf{u}=\mathbf{g}(\mathbf{x}),\qquad\mathbf{y}=\mathbf{f}(\mathbf{u}).$$

Then the scalar multiplication of derivatives becomes **matrix multiplication of Jacobians**:

:::def Vector chain rule
$$\frac{\partial\mathbf{y}}{\partial\mathbf{x}}=\frac{\partial\mathbf{y}}{\partial\mathbf{u}}\,\frac{\partial\mathbf{u}}{\partial\mathbf{x}}$$

Shape check:

$$(m\times k)(k\times n)=m\times n.$$
:::

:::fig vector-chain-graph
:::

:::ml Backpropagation
A neural network is a computation graph. Backpropagation starts from the scalar loss and repeatedly applies the chain rule **backward**, multiplying by each local derivative. Modern libraries do this automatically; matrix calculus explains what the autograd engine is computing.
:::

For a simple neuron

$$z=\mathbf{w}^T\mathbf{x}+b,\qquad a=\sigma(z),\qquad L=L(a),$$

the parameter gradient follows the chain

$$\boxed{\nabla_{\mathbf{w}}L=\frac{\partial L}{\partial a}\,\frac{\partial a}{\partial z}\,\mathbf{x}}$$

(using the common column-gradient ML notation here). The important idea is not memorizing this one expression: it is **local derivatives multiplied through the graph**.

## 1.3.5 Matrix Calculus Identities

:::add Added note
The Parr–Howard paper derives the Jacobian, element-wise, scalar-expansion, reduction, dot-product, and chain-rule identities above. The following compact table also includes standard identities commonly needed immediately afterward in ML derivations.
:::

For this table, $\nabla_{\mathbf{x}}f$ is written in the common ML **column-gradient** convention. $A$ and vectors not being differentiated are treated as constants.

|{stack}
| **Expression** | **Derivative / gradient** |
|---|---|
| $c$ | $\nabla_{\mathbf{x}}c=\mathbf{0}$ |
| $\mathbf{a}^T\mathbf{x}$ | $\nabla_{\mathbf{x}}(\mathbf{a}^T\mathbf{x})=\mathbf{a}$ |
| $\mathbf{x}^T\mathbf{x}=\|\mathbf{x}\|_2^2$ | $\nabla_{\mathbf{x}}(\mathbf{x}^T\mathbf{x})=2\mathbf{x}$ |
| $\mathbf{x}^TA\mathbf{x}$ | $\nabla_{\mathbf{x}}(\mathbf{x}^TA\mathbf{x})=(A+A^T)\mathbf{x}$ |
| $\mathbf{x}^TA\mathbf{x},\ A=A^T$ | $\nabla_{\mathbf{x}}(\mathbf{x}^TA\mathbf{x})=2A\mathbf{x}$ |
| $\|A\mathbf{x}-\mathbf{b}\|_2^2$ | $\nabla_{\mathbf{x}}=2A^T(A\mathbf{x}-\mathbf{b})$ |
| $\mathbf{y}=A\mathbf{x}+\mathbf{b}$ | $J_{\mathbf{x}}\mathbf{y}=A$ |
| $\mathbf{y}=g(\mathbf{x})$ element-wise | $J=\operatorname{diag}\big(g'(x_1),\dots,g'(x_n)\big)$ |
| $\mathbf{a}^TW\mathbf{b}$ | $\nabla_W(\mathbf{a}^TW\mathbf{b})=\mathbf{a}\mathbf{b}^T$ |
| $\operatorname{tr}(W^TA)$ | $\nabla_W\operatorname{tr}(W^TA)=A$ |
| $\|W\|_F^2=\sum_{ij}W_{ij}^2$ | $\nabla_W\|W\|_F^2=2W$ |

### Why the quadratic-loss identity matters

Least squares uses

$$L(\mathbf{x})=\|A\mathbf{x}-\mathbf{b}\|_2^2.$$

Its gradient is

$$\boxed{\nabla_{\mathbf{x}}L=2A^T(A\mathbf{x}-\mathbf{b})}$$

The term $(A\mathbf{x}-\mathbf{b})$ is the **residual/error**; multiplying by $A^T$ maps that output-space error back to parameter space. This is the squared $L_2$ norm from §1.1.7 being used as a loss.

### Trace trick

A scalar can always be written as its own trace, and trace lets us cyclically rotate matrix products:

$$\boxed{\operatorname{tr}(ABC)=\operatorname{tr}(BCA)=\operatorname{tr}(CAB)}$$

This is useful for rewriting scalar losses into a form where the matrix being differentiated is easy to isolate.

:::ml Practical rule
If a matrix derivative looks terrifying, reduce the expression to **scalar entries**, differentiate those with ordinary partial-derivative rules, and then reassemble the vector/matrix. This is also the strategy emphasized in the matrix-calculus paper.
:::

## 1.3.6 Convexity & Optimization

:::add Added note
Convexity is part of the course outline but is outside the scope of the Parr–Howard paper. This subsection gives the standard ML background needed to interpret loss landscapes and optimization.
:::

### Optimization problem

Training usually means

$$\boxed{\min_{\boldsymbol\theta}L(\boldsymbol\theta)}$$

where $\boldsymbol\theta$ contains the model parameters.

### Gradient descent

Because $\nabla L$ points uphill, move in the opposite direction:

:::def Gradient descent update
$$\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta\,\nabla L(\boldsymbol\theta_t)$$

$\eta>0$ is the **learning rate** (step size).
:::

:::fig gd-eta
:::

- $\eta$ too small $\to$ training is slow.
- $\eta$ too large $\to$ overshooting/instability is possible.
- $\nabla L=\mathbf{0}\to$ a **stationary point**, but not automatically a minimum.

### Second derivatives and the Hessian

For one variable, at a stationary point:

$$f''(x)>0\;\Rightarrow\;\text{local minimum},\qquad\qquad f''(x)<0\;\Rightarrow\;\text{local maximum}.$$

For many variables, collect second partial derivatives into the **Hessian**:

:::def Hessian
$$H_f(\mathbf{x})=\begin{bmatrix}\dfrac{\partial^2f}{\partial x_1^2}&\cdots&\dfrac{\partial^2f}{\partial x_1\partial x_n}\\\vdots&\ddots&\vdots\\\dfrac{\partial^2f}{\partial x_n\partial x_1}&\cdots&\dfrac{\partial^2f}{\partial x_n^2}\end{bmatrix}$$

The Hessian describes **local curvature**.
:::

At a stationary point, roughly:

- $H\succ0$ (positive definite) $\to$ curves upward in every direction $\to$ local **minimum**.
- $H\prec0\to$ curves downward in every direction $\to$ local **maximum**.
- mixed positive/negative curvature $\to$ **saddle point**.

Positive definite just means all the Hessian's eigenvalues are positive (§1.1.5): every eigen-direction curves up.

:::viz hessian | Set the Hessian entries. The surface is $f=\tfrac12\mathbf{x}^TH\mathbf{x}$ around a stationary point; its eigen-directions are coloured blue where they curve up and red where they curve down.
:::

### Convex functions

:::def Convexity
A function $f$ is convex if, for any $\mathbf{x},\mathbf{y}$ and $0\le\lambda\le1$,

$$f\big(\lambda\mathbf{x}+(1-\lambda)\mathbf{y}\big)\le\lambda f(\mathbf{x})+(1-\lambda)f(\mathbf{y}).$$

Geometrically: the straight line joining two points on the graph never falls below the function.
:::

For twice-differentiable $f$,

$$\boxed{f\text{ convex}\iff H_f(\mathbf{x})\succeq0\text{ everywhere.}}$$

A **convex set** is a set with no dents: for any two points in it, the whole line segment between them stays inside. (A convex function needs a convex domain; in constrained optimization, a convex constraint region is what keeps "local = global" alive.)

**First-order test.** For differentiable $f$, convexity is equivalent to

$$f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x})\qquad\text{for all }\mathbf{x},\mathbf{y}$$

— the graph sits **above every tangent**. So any point with $\nabla f=\mathbf{0}$ (a flat tangent) is instantly a **global** minimum. That is the convexity magic in one line.

:::fig convex-nonconvex
:::

:::viz convexity | Drag $\mathbf{x}$ and $\mathbf{y}$ and slide $\lambda$: the chord value $\lambda f(\mathbf{x})+(1-\lambda)f(\mathbf{y})$ against $f(\lambda\mathbf{x}+(1-\lambda)\mathbf{y})$, and the tangent at $\mathbf{x}$. Switch to the non-convex curve to see both tests fail.
:::

:::ml Why convexity matters
In a convex problem, **every local minimum is global**, so optimization is much easier to reason about. Linear regression with squared loss is convex. Deep neural networks are generally **non-convex**, so their landscapes can contain saddles and many different minima.
:::

### Loss landscapes and contours

For two parameters $w_1,w_2$, a loss such as

$$L(w_1,w_2)=w_1^2+w_2^2$$

is a 3D bowl. A contour plot is the bird's-eye view (same picture as §1.1.7): each contour contains parameter values with the same loss. The negative gradient points **perpendicular** to the contour toward decreasing loss.

:::ex Simple optimization example
For $L(\mathbf{w})=\mathbf{w}^T\mathbf{w}$, the gradient is $\nabla L=2\mathbf{w}$. Gradient descent gives

$$\mathbf{w}_{t+1}=\mathbf{w}_t-2\eta\mathbf{w}_t=(1-2\eta)\,\mathbf{w}_t,$$

so, for a suitable step size, the weights repeatedly shrink toward the minimum $\mathbf{w}=\mathbf{0}$.\
The "suitable" step size here is $0<\eta<1$, so that $|1-2\eta|<1$ and each step really does shrink $\mathbf{w}$. With $\eta>1$, the factor is below $-1$ and the weights flip sign and grow: exactly the overshooting picture above.
:::

:::viz gd | Slide $\eta$ and run gradient descent on $L(\mathbf{w})=\mathbf{w}^T\mathbf{w}$. Each step multiplies $\mathbf{w}$ by $1-2\eta$; the arrows are $-\eta\nabla L$, perpendicular to the contours.
:::

## 1.3.7 Quick Reference

|{stack}
| **Concept** | Formula / definition | Meaning |
|---|---|---|
| **Limit** | $\lim_{x\to a}f(x)=L$ | what $f$ approaches near $a$ |
| **Derivative** | $f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$ | local slope / sensitivity |
| **Local linearization** | $\Delta f\approx f'(x)\Delta x$ | tiny input change $\to$ approx. output change |
| **Partial derivative** | $\partial f/\partial x_i$ | sensitivity to one variable, others fixed |
| **Gradient** | $\nabla f=[\partial f/\partial x_1,\dots]$ | all partials of one scalar output |
| **Jacobian** | $J=\partial\mathbf{y}/\partial\mathbf{x}$ | all output-vs-input sensitivities |
| **Jacobian shape** | $m\times n$ | $m$ outputs, $n$ inputs |
| **Identity derivative** | $\partial\mathbf{x}/\partial\mathbf{x}=I$ | each $x_i$ only depends on itself |
| **Element-wise op.** | off-diagonal partials $=0$ | Jacobian becomes diagonal |
| **Scalar expansion** | $\partial(z\mathbf{x})/\partial\mathbf{x}=zI$ | scalar copied across vector |
| **Vector sum** | $\partial\sum_ix_i/\partial\mathbf{x}=\mathbf{1}^T$ | every element contributes once |
| **Dot product** | $\partial(\mathbf{w}^T\mathbf{x})/\partial\mathbf{x}=\mathbf{w}^T$ | weighted sum sensitivity |
| **Scalar chain rule** | $\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}$ | multiply local derivatives |
| **Total derivative** | direct path $+$ all indirect paths | sum every route from input to output |
| **Vector chain rule** | $J_{f\circ g}=J_fJ_g$ | chain rule becomes matrix multiplication |
| **Quadratic form** | $\nabla_{\mathbf{x}}(\mathbf{x}^TA\mathbf{x})=(A+A^T)\mathbf{x}$ | core matrix identity |
| **Squared residual** | $\nabla\|A\mathbf{x}-\mathbf{b}\|^2=2A^T(A\mathbf{x}-\mathbf{b})$ | least-squares gradient |
| **Gradient descent** | $\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta\nabla L$ | move opposite steepest increase |
| **Hessian** | $H_{ij}=\partial^2f/(\partial x_i\partial x_j)$ | local curvature |
| **Convex** | $f(\lambda\mathbf{x}+(1-\lambda)\mathbf{y})\le\lambda f(\mathbf{x})+(1-\lambda)f(\mathbf{y})$ | local minima are global |
| **First-order test** | $f(\mathbf{y})\ge f(\mathbf{x})+\nabla f(\mathbf{x})^T(\mathbf{y}-\mathbf{x})$ | graph sits above every tangent |
| **Convexity test** | $H\succeq0$ everywhere | nonnegative curvature in all directions |

:::def Big picture
In ML, the whole part collapses to one storyline:

$$\text{loss }L(\boldsymbol\theta)\to\text{gradient }\nabla L\to\text{chain rule/backprop}\to\text{update parameters to reduce }L.$$

Everything else in this section is machinery that makes that computation possible for vectors and matrices.
:::

> **Source basis.** Sections on partial derivatives, gradients, Jacobians, element-wise operations, scalar expansion, vector reductions, and chain rules follow Terence Parr and Jeremy Howard, *The Matrix Calculus You Need For Deep Learning* (2018). Items marked "Added note" provide standard background required by the MLR 570 section outline but not developed in that paper.
