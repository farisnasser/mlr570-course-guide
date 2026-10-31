:::plain
**The flow:** Matrix → Determinant → Det = 0 → Rank → Rank–Nullity → Null Space → Diagonal matrices → Eigenvectors → Diagonalization → Norms → SVD.
:::

## 1.1.1 Matrices as Linear Transformations

### Matrix

A matrix $\begin{bmatrix}a&b\\c&d\end{bmatrix}$ is a **linear transformation**. It acts on the basis vectors

$$e_1=\begin{bmatrix}1\\0\end{bmatrix},\qquad e_2=\begin{bmatrix}0\\1\end{bmatrix}.$$

:::fig grid-before-after
:::

### Determinant

So applying $\begin{bmatrix}a&b\\c&d\end{bmatrix}$ tells us how much the grid is stretched by.

:::def Determinant
This stretching factor applies to the *whole grid*; it is called $|\det(A)|$.
:::

:::add Added note
For a $2\times2$ matrix: $\det\begin{bmatrix}a&b\\c&d\end{bmatrix}=ad-bc$. The sign tells you whether the orientation flipped; the absolute value is the area stretch factor.
:::

### Case: $\det(A)=0$

This means the transformed basis directions fail to span the full space. **Hence, a dimension was lost.**

:::def Why $\det=0$ means not invertible
This is why a matrix with $\det=0$ is **not invertible**: if a transformation collapses several different original points onto the same output, then, given the output, we can't determine which original point it came from.
:::

:::viz grid-transform | Drag the tips of the two columns $\begin{bmatrix}a\\c\end{bmatrix}$ and $\begin{bmatrix}b\\d\end{bmatrix}$, or pick a matrix from the notes; the slider plays the transformation.
:::

### Multiplying matrices = chaining transformations

The product $AB$ means: apply $B$ **first**, then $A$. For $A\in\mathbb{R}^{m\times n}$ and $B\in\mathbb{R}^{n\times p}$,

$$C=AB\in\mathbb{R}^{m\times p},\qquad c_{ij}=\sum_k a_{ik}b_{kj}$$

(each entry combines a row of $A$ with a column of $B$). It only works when the **inner dimensions match**: columns of $A$ = rows of $B$.

:::def Order matters
In general $AB\neq BA$: transformations don't commute. Stretching then rotating is not the same grid as rotating then stretching.
:::

:::viz chain | Set the stretch and the rotation; the left grid stretches then rotates, the right grid rotates then stretches.
:::

### Transpose

$(A^T)_{ij}=A_{ji}$: flip the matrix across its main diagonal, so rows become columns.

$$A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}\quad\Rightarrow\quad A^T=\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}$$

Rules: $(A^T)^T=A$, $\ (A+B)^T=A^T+B^T$, and the one to be careful with:

$$\boxed{(AB)^T=B^TA^T}$$

The order **reverses**. If $A^T=A$, the matrix is **symmetric** (more on it in §1.1.4).

### Matrix inverse

The inverse is the transformation that **undoes** $A$:

$$AA^{-1}=A^{-1}A=I.$$

:::def When does it exist?
A square $A$ is invertible $\iff\det(A)\neq0\iff$ full rank. Same story as above: if $\det=0$, a dimension collapsed, several inputs landed on the same output, and no map can tell them apart again.
:::

For a $2\times2$ matrix there is a direct formula:

$$A=\begin{bmatrix}a&b\\c&d\end{bmatrix}\quad\Rightarrow\quad A^{-1}=\frac{1}{\det A}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$$

:::ex Example
$A=\begin{bmatrix}4&7\\2&6\end{bmatrix}$: $\ \det A=24-14=10\neq0$, so

$$A^{-1}=\frac{1}{10}\begin{bmatrix}6&-7\\-2&4\end{bmatrix}.\qquad\text{Check: }AA^{-1}=\frac{1}{10}\begin{bmatrix}10&0\\0&10\end{bmatrix}=I.\ \checkmark$$

But $\begin{bmatrix}2&4\\1&2\end{bmatrix}$ has $\det=0\to$ singular, no inverse.
:::

Like the transpose, the inverse of a product **reverses the order**: $(AB)^{-1}=B^{-1}A^{-1}$ — to undo, undo the last transformation first. Also $(A^T)^{-1}=(A^{-1})^T$.

:::add Added note
Only square, full-rank matrices have a true inverse. A rectangular matrix never does; the closest substitute is the Moore–Penrose **pseudoinverse**, built from the SVD.
:::

This leads us to **Rank**.

## 1.1.2 Rank

:::def Rank
Rank is the # of independent directions represented by a matrix.

- A $3\times3$ matrix has rank $=3$ if it preserves all 3 dimensions.

So $\to$ **Rank = # of independent dims that survive.**
:::

### Column space: what rank actually measures

Multiplying $A\mathbf{x}$ is really **mixing the columns** of $A$:

$$A\mathbf{x}=\begin{bmatrix}|&&|\\\mathbf{a}_1&\cdots&\mathbf{a}_n\\|&&|\end{bmatrix}\begin{bmatrix}x_1\\\vdots\\x_n\end{bmatrix}=x_1\mathbf{a}_1+\cdots+x_n\mathbf{a}_n.$$

So the set of every output $A$ can produce is the set of all mixes of its columns.

:::def Column space
$$\operatorname{Col}(A)=\{\,A\mathbf{x}\,\}=\text{span of the columns of }A$$

It is everything the matrix can **reach**. And this is where rank formally lives:

$$\boxed{\operatorname{rank}(A)=\dim\operatorname{Col}(A)=\#\text{ of linearly independent columns}}$$

The row space has the same dimension — that's why counting independent rows or independent columns gives the same rank.
:::

:::ex Example
$$L=\begin{bmatrix}1&0&2&3&4\\0.5&0&1&1.5&2\\2&0&4&6&8\\1&0&2&3&4\\2&0&4&6&8\end{bmatrix}\qquad\Rightarrow\qquad\operatorname{rank}(L)=1.$$

Every row is a multiple of the first ($R_2=\tfrac12R_1$, $R_3=2R_1$, $R_4=R_1$, $R_5=2R_1$), so the whole matrix carries **one** pattern of info.
:::

### Solving $A\mathbf{x}=\mathbf{b}$

A system of linear equations is the question: *which mix of the columns of $A$ gives $\mathbf{b}$?*

$$\boxed{A\mathbf{x}=\mathbf{b}\text{ has a solution}\iff\mathbf{b}\in\operatorname{Col}(A)}$$

Three things can happen:

:::ex The three cases
**1) Unique solution:** $\ \ x+y=3,\ 2x-y=0$. Eliminate: $R_2-2R_1\Rightarrow-3y=-6\Rightarrow y=2,\ x=1$. The lines cross at one point.

**2) No solution:** $\ \ x+y=2,\ 2x+2y=5$. Eliminate: $R_2-2R_1\Rightarrow0=1$, a contradiction. Parallel lines; $\mathbf{b}$ is outside $\operatorname{Col}(A)$.

**3) Infinitely many:** $\ \ x+2y=4,\ 2x+4y=8$. The second equation is just $2\times$ the first. Let $y=t$: $\mathbf{x}=(4-2t,\ t)$ for any $t$ — a whole line of solutions.

:::fig three-cases
:::
:::

:::def Rank decides
For a square $A$: full rank $\Rightarrow$ exactly one solution, $\mathbf{x}=A^{-1}\mathbf{b}$. Rank-deficient $\Rightarrow$ no solution ($\mathbf{b}$ outside $\operatorname{Col}(A)$) or infinitely many ($\mathbf{b}$ inside, plus anything from the null space, §1.1.3).
:::

:::viz solve | Pick one of the three cases, then drag $\mathbf{b}$; the left plot shows the two lines, the right plot shows $\operatorname{Col}(A)$ and $\mathbf{b}$.
:::

### From a data perspective

:::cols figleft
:::col
:::fig data-matrix
:::
:::
:::col
Each column is a feature (an axis), each row is a sample.\
**Rank = # of independent patterns of info present** (unique info).
:::
:::

:::ml Rank in ML
1. **Preventing redundancy.** If two feature columns are exact copies or multiples of each other, they don't add new info. Hence rank $\downarrow$.
2. **Checking model solvability.** If the matrix is full rank, the data has no useless duplication. The model trains stably.
3. **Data compression and noise reduction.** Many real-world datasets have low rank because features overlap heavily. **PCA & SVD find low-rank approximations.**
:::

### Rule: maximum rank

:::def Rule
$$\text{Max rank}=\min(m,n)$$

The rank of a matrix can never be larger than its smallest dimension.
:::

:::ex Examples
- Data on 10,000 houses (rows), 3 features (cols). Dimension: $10{,}000\times3$, so max possible rank $=3$.
- In hopeful situations: let's say we have 50 features (columns) and 10,000 samples (rows) $\to$ max rank is 50. In a healthy dataset, the calculated rank is exactly $=50$.
- 1,000 houses (rows), 2 features (cols): square meters ($X$) & bedrooms ($Y$). Max rank $=2$.

:::cols figleft
:::col
:::fig houses-scatter
:::
:::
:::col
If we plot the data: points are only plotted on 2D because we don't have a third feature — we are restricted to 2-D permanently.
:::
:::
:::

:::ml Low rank: curse or blessing?
**Curse:** a low-rank square matrix is singular — $\det=0$, no inverse, and $A\mathbf{x}=\mathbf{b}$ has no solution or infinitely many.\
**Blessing:** low rank means the matrix can be rebuilt from far fewer numbers. If $W\in\mathbb{R}^{1000\times1000}$ has rank 5, every row/column lives in a 5-dimensional subspace, so it factors as

$$W=UV,\qquad U\in\mathbb{R}^{1000\times5},\quad V\in\mathbb{R}^{5\times1000}.$$

Storage drops from $10^6$ numbers to 10,000, and $W\mathbf{x}$ becomes two cheap products $U(V\mathbf{x})$. Low rank = redundancy, and exploiting it is exactly what PCA & SVD compression do. The price: a rank-limited matrix can't represent rich relationships — cap the rank too hard and the model underfits.
:::

## 1.1.3 Rank–Nullity and the Null Space

### Rank–Nullity

:::def Rank–Nullity Theorem
$$\text{rank}+\text{nullity}=n\qquad\text{with }n\text{ input dimensions.}$$

The input space splits into:

- directions that **survive** (rank),
- directions that **disappear** (nullity).
:::

### Null space

:::def Null space
$$\operatorname{Null}(A)=\{\,x\ \ \text{s.t.}\ \ Ax=0\,\}$$

The null space tells us exactly what information the matrix is **blind to**.
:::

:::ex Example
$A=\begin{bmatrix}1&2\\2&4\end{bmatrix}$.

$$Ax=0\;\Rightarrow\;\begin{bmatrix}1&2\\2&4\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix}=0\;\Rightarrow\;x_1+2x_2=0\;\Rightarrow\;x_1=-2x_2$$

Let $x_2=t$:

$$x=\begin{bmatrix}-2t\\t\end{bmatrix}=t\begin{bmatrix}-2\\1\end{bmatrix}$$

:::cols
:::col
So any vector in the direction $\begin{bmatrix}-2\\1\end{bmatrix}$ is **erased** by $A$.\
The null space here is a line, so nullity (dimension) $=1$.\
The null space basically means it crushes anything that enters its "space".
:::
:::col
:::fig null-line
:::
:::
:::
:::

:::add Added note
Check with rank–nullity: the second row of $A$ is $2\times$ the first, so rank $=1$. Then rank $+$ nullity $=1+1=2=n$. ✓
:::

:::ex Nullity can be bigger than 1
$A=\begin{bmatrix}1&2&-1\\2&4&-2\end{bmatrix}$: rank $=1$ (second row $=2\times$ first), $n=3$, so nullity $=2$. Solving $x_1+2x_2-x_3=0$ gives a whole **plane** of erased vectors, e.g. spanned by $(1,0,1)^T$ and $(0,1,2)^T$.
:::

:::ml Null space in ML
The null space represents the **blind spot** of an ML model. It contains the directions in which your data *can* change, but the model will not see it.

- If a vector $z$ is in the null space of matrix $A$, then multiplying them gives 0: $Az=0$.
- Due to that, we can add this null-space noise $z$ to your original data $x$, and the model output is exactly the same:
   $$A(x+z)=Ax+Az=Ax+0=Ax$$
- To the ML model, $x$ and $(x+z)$ are **identical**.

**Why it matters in ML:**

- **PCA:** when compressing data, we intentionally look for directions in the data that don't contribute much to the final prediction.
- **Rank–nullity balancing:** rank $+$ nullity $=$ total features, e.g.
   $$\underbrace{7}_{\text{useful features}}\quad+\quad\underbrace{3}_{\text{dead zones, data change ignored}}\quad=10\text{ features}$$
:::

:::viz nullspace3d | Drag in the left view to orbit. Move $x$ with the sliders: moving inside the red plane changes $x$ but not $Ax$.
:::

## 1.1.4 Common Types of Matrices

### Diagonal matrix

$$D=\begin{bmatrix}3&0&0\\0&7&0\\0&0&2\end{bmatrix},\qquad x=\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}\qquad\Rightarrow\qquad Dx=\begin{bmatrix}3x_1\\7x_2\\2x_3\end{bmatrix}$$

$D$ simply says: stretch direction 1 by 3, direction 2 by 7, direction 3 by 2. Every direction is handled on its own — nothing gets mixed.

This becomes important with the **eigendecomposition**: we want to find a coordinate system in which a complicated matrix behaves like a diagonal matrix. We look for:

1. **eigenvectors:** brand-new axes (directions),
2. **eigenvalues:** scale.

:::ml Diagonalization in ML
It is the ultimate clean-up tool for ML data. It takes a messy dataset where features are tangled up and alters perspectives so that every feature becomes perfectly independent.
:::

:::add Added note
Strictly, diagonalizing the data's *covariance* matrix makes the new features *uncorrelated* (this is exactly what PCA does). "Independent" is the intuition; "uncorrelated" is the precise term.
:::

### Other types worth recognizing

**Identity matrix $I$:** ones on the diagonal, zeros elsewhere. $AI=IA=A$ — the "do nothing" transformation.

**Symmetric matrix:** $A^T=A$.

$$A=\begin{bmatrix}2&-1&0\\-1&3&4\\0&4&5\end{bmatrix}$$

The matrices ML cares about most are symmetric: **covariance matrices** and **Hessians**. Symmetry comes with two gifts: the eigenvalues are always **real**, and the eigenvectors can be chosen **orthonormal**, so

$$A=Q\Lambda Q^T$$

— diagonalization where the inverse is free, because $Q^{-1}=Q^T$ (next type).

**Orthogonal / orthonormal matrix:** $Q^TQ=I$. Columns are mutually perpendicular and unit length. For a square $Q$ this forces

$$\boxed{Q^{-1}=Q^T}$$

the cheapest inverse there is. These matrices **preserve lengths and angles**: pure rotations/reflections, no stretching.

:::ex Example
$$Q=\frac{1}{\sqrt2}\begin{bmatrix}1&1\\-1&1\end{bmatrix}\qquad Q^TQ=I,$$

and geometrically $Q$ rotates the plane by $45^\circ$.
:::

This is exactly what $U$ and $V$ are in the SVD (§1.1.8): the two "rotate" steps are orthogonal matrices, and $V^{-1}=V^T$ is why the formula ends with $V^T$.

**Rank-one matrix:** $A=\mathbf{u}\mathbf{v}^T$ (outer product of two vectors):

$$\mathbf{u}=\begin{bmatrix}1\\2\end{bmatrix},\quad\mathbf{v}=\begin{bmatrix}3\\4\\5\end{bmatrix}\quad\Rightarrow\quad\mathbf{u}\mathbf{v}^T=\begin{bmatrix}3&4&5\\6&8&10\end{bmatrix}.$$

Every row is a multiple of $\mathbf{v}^T$, so rank $=1$: one pattern of info. The SVD layers $\sigma_i\mathbf{u}_i\mathbf{v}_i^T$ (§1.1.8) are exactly these.

|{stack}
| **Type** | Definition | Where it shows up |
|---|---|---|
| **Skew-symmetric** | $A^T=-A$ (zero diagonal) | rotations/angular momentum in physics; purely imaginary eigenvalues |
| **Upper/lower triangular** | zeros below/above the diagonal | LU decomposition; solving systems by forward/backward substitution |
| **Sparse** | most entries are 0 | recommender systems, bag-of-words, graph adjacency; cheap storage |

## 1.1.5 Eigenvectors and Eigenvalues

A matrix is a transformation: for $Ax$, applying $A$ to most vectors would change their **length and direction**.

**However! Eigenvector directions do not change.**

:::def Definition
A nonzero vector $v$ is an eigenvector of $A$ if

$$Av=\lambda v$$

Multiplying $v$ by $A$ would give $v$ itself, just stretched by the scalar $\lambda$ (the eigenvalue).
:::

### Why $\det(A-\lambda I)=0$?

$$Av=\lambda v\;\to\;Av-\lambda v=0\;\to\;Av-\lambda Iv=0\;\to\;\boxed{(A-\lambda I)v=0}$$

- We don't want $v=0$, since eigenvectors must be non-zero.
- So, for what values of $\lambda$ does the matrix $A-\lambda I$ have a nonzero vector in its null space?
- If it has a nonzero solution, then $A-\lambda I$ has a non-trivial null space. Hence it is **not full rank**. It is **not invertible**. Therefore:
   $$\boxed{\det(A-\lambda I)=0}$$

:::add Added note
This is the whole chain of the notes connecting: non-trivial null space $\Rightarrow$ not full rank $\Rightarrow$ a dimension was lost $\Rightarrow$ $\det=0$ (§1.1.1).
:::

:::viz eigen | Drag $v$ around the circle and watch $Av$; the slider plays the transformation. The amber lines are the eigenvector directions.
:::

:::ml Importance in ML
- Finding the true shape of data.
- Eigenvectors point at the highest variation in data (most dominant trend).
:::

:::add Added note
Precisely: the eigenvectors of the data's **covariance matrix** point along the directions of variation, and the one with the **largest eigenvalue** is the direction of highest variation. That's the first principal component in PCA.
:::

## 1.1.6 Diagonalization

:::def Diagonalization
Switching the matrix's basis to a basis made of its eigenvectors.\
Diagonalization means switching to the eigenvector directions as our coordinate system.
:::

:::cols
:::col
**Intuition:** imagine a mechanics problem — we switch from the horizontal and vertical basis to *along the ramp* and *perpendicular to the ramp*. The problem is the same, but in the right axes it becomes simple.
:::
:::col
:::fig ramp
:::
:::
:::

### Building $P$ and $D$

Take $A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$ with eigens

$$v_1=\begin{bmatrix}1\\1\end{bmatrix},\ \lambda_1=3\qquad\qquad v_2=\begin{bmatrix}1\\-1\end{bmatrix},\ \lambda_2=1$$

- $P=\begin{bmatrix}1&1\\1&-1\end{bmatrix}$ — just placing the eigenvectors into a matrix (as columns).
- $D=\begin{bmatrix}3&0\\0&1\end{bmatrix}$ — place their corresponding eigenvalues in another (diagonal) matrix.

:::def Eigendecomposition
$$A=PDP^{-1}$$

- $P$: has the eigenvectors as columns.
- $D$: diagonal, with the corresponding eigenvalues.
:::

### Explanation of $A=PDP^{-1}$

Read it **from right to left**:

1. $P^{-1}$: change the matrix into its natural directions (eigenvector axes).
2. $D$: perform scaling.
3. $P$: change back to the regular directions.

**Diagonal matrices make it easy for $D^k$ to be computed:**

$$D=\begin{bmatrix}4&0\\0&3\end{bmatrix}\quad\Rightarrow\quad D^{10}=\begin{bmatrix}4^{10}&0\\0&3^{10}\end{bmatrix}$$

So the idea is to take matrix $A$ and factor it: $A=PDP^{-1}$.

:::ex e.g. computing $A^4$
$$A^4=\big(PDP^{-1}\big)\big(PDP^{-1}\big)\big(PDP^{-1}\big)\big(PDP^{-1}\big)=PD\underbrace{P^{-1}P}_{I}D\underbrace{P^{-1}P}_{I}D\underbrace{P^{-1}P}_{I}DP^{-1}$$

$$\boxed{A^4=PD^4P^{-1}}$$
:::

:::viz diagonalize | Step through $P^{-1}$, $D$, $P$ (right to left); the last frame is exactly $A$. The power slider shows $A^k=PD^kP^{-1}$.
:::

### Worked example

:::ex E.g. $A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$
**Step 1) Find eigenvalues:** $\det(A-\lambda I)=0$

$$A-\lambda I=\begin{bmatrix}2&1\\1&2\end{bmatrix}-\begin{bmatrix}\lambda&0\\0&\lambda\end{bmatrix}=\begin{bmatrix}2-\lambda&1\\1&2-\lambda\end{bmatrix}$$

$$(2-\lambda)^2-1=0\qquad\Rightarrow\qquad\lambda_1=3,\quad\lambda_2=1$$

**Step 2) Now find the corresponding eigenvectors:**

:::cols even
:::col
• For $\lambda_1=3$: $(A-3I)v=0$

$$\begin{bmatrix}-1&1\\1&-1\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}=0$$

$-x+y=0\Rightarrow\boxed{x=y}$\
So any vector where $y=x$ works:\
$\begin{bmatrix}1\\1\end{bmatrix},\begin{bmatrix}2\\2\end{bmatrix},\begin{bmatrix}3\\3\end{bmatrix},\ldots$ They all point in the same direction, so just take the simplest: $v_1=\begin{bmatrix}1\\1\end{bmatrix}$.
:::
:::col
• For $\lambda_2=1$: $(A-I)v=0$

$$\begin{bmatrix}1&1\\1&1\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}=0$$

$x+y=0\Rightarrow x=-y$\
Choose $v_2=\begin{bmatrix}1\\-1\end{bmatrix}$.
:::
:::

**Step 3) Assemble:**

$$P=\begin{bmatrix}1&1\\1&-1\end{bmatrix},\qquad D=\begin{bmatrix}3&0\\0&1\end{bmatrix},\qquad\boxed{A=PDP^{-1}}$$
:::

:::add Added note
**Verification.** $P^{-1}=\tfrac{1}{-2}\begin{bmatrix}-1&-1\\-1&1\end{bmatrix}=\tfrac12\begin{bmatrix}1&1\\1&-1\end{bmatrix}$. Then

$$PD=\begin{bmatrix}3&1\\3&-1\end{bmatrix},\qquad PDP^{-1}=\tfrac12\begin{bmatrix}3&1\\3&-1\end{bmatrix}\begin{bmatrix}1&1\\1&-1\end{bmatrix}=\tfrac12\begin{bmatrix}4&2\\2&4\end{bmatrix}=\begin{bmatrix}2&1\\1&2\end{bmatrix}=A.\ \checkmark$$

The order of columns in $P$ must match the order of eigenvalues in $D$.
:::

## 1.1.7 Norms

:::def Norm
A norm measures the **size** of a vector/matrix, e.g. of $x=\begin{bmatrix}x_1\\\vdots\\x_n\end{bmatrix}$.
:::

:::ml Norms in ML
We use norms to measure errors, compare vectors, and decide whether the model is overfitting/generalizing.
:::

### $L_2$ norm (Euclidean)

$$\|x\|_2=\sqrt{x_1^2+x_2^2+\cdots+x_n^2}$$

- Straight-line distance (Euclidean) from the origin to the point $x=\begin{bmatrix}x_1\\x_2\end{bmatrix}$.
- Distance between two points: $\|x-y\|_2$.

**Why a circle?** We ask: which points have exactly the same norm? All points on a circle are exactly the same distance away from the centre.

:::ml $L_2$ regularization
Perfect for keeping all weights **small but not zero**.
:::

### $L_1$ norm (Manhattan distance)

$$\|x\|_1=\sum_i|x_i|\qquad\text{e.g. }x=\begin{bmatrix}3\\4\end{bmatrix}:\quad\|x\|_1=|3|+|4|=7$$

Useful: imagine only being able to travel vertically or horizontally.

:::add Added note
Same vector, different norms: $\|(3,4)\|_2=\sqrt{9+16}=5$, while $\|(3,4)\|_1=7$.
:::

:::ml $L_1$ in ML
Important for **sparse models**, since the $L_1$ norm drives many weights to zero, performing **feature selection**.\
**Goal:**

$$\min_w\Big[\quad\underbrace{L(w)}_{\substack{\text{loss: diff between}\\\text{actual vs predicted}}}\quad+\quad\underbrace{\lambda\,\|w\|_1}_{\substack{\text{charges the model for}\\\text{using large weights}}}\quad\Big]$$

$\lambda$ controls how much we care.
:::

### $L_\infty$, Frobenius, and what makes a norm

**$L_\infty$ norm (max norm):** only the biggest coordinate matters:

$$\|x\|_\infty=\max_i|x_i|$$

:::ex One vector, three sizes
$$x=[3,-4]:\qquad\|x\|_1=3+4=7,\qquad\|x\|_2=\sqrt{9+16}=5,\qquad\|x\|_\infty=4.$$
:::

**Frobenius norm (for matrices):** flatten the matrix into one long vector and take its $L_2$ norm:

$$\|A\|_F=\sqrt{\sum_{i,j}a_{ij}^2}\qquad\text{e.g.}\quad A=\begin{bmatrix}1&2\\3&4\end{bmatrix}:\ \|A\|_F=\sqrt{1+4+9+16}=\sqrt{30}.$$

This is the $\|W\|_F^2$ that shows up when regularizing weight matrices (§1.3.5).

:::add Added note
Every norm obeys three rules: $\|x\|\ge0$, with $\|x\|=0$ only for $x=0$; $\ \|\alpha x\|=|\alpha|\,\|x\|$; and the triangle inequality $\|x+y\|\le\|x\|+\|y\|$ — a detour is never shorter than the direct path.
:::

### Shapes of equal norm

:::fig norm-shapes
:::

- **The plot of $L_1$ norms:** contours in a tilted plane (a diamond) — points that all lie equally distant from the origin. For $L_2$ it is circular.
- **Ellipses:** some directions are more expensive than others.
   $$x^TAx=1\qquad\text{e.g. }x_1^2+4x_2^2=1$$
   Simply a circle, stretched.

:::add Added note
The ellipse example is $x^TAx=1$ with $A=\begin{bmatrix}1&0\\0&4\end{bmatrix}$. Moving in the $x_2$ direction costs $4\times$ more, so the curve only reaches $x_2=\pm\frac12$ but $x_1=\pm1$. With a non-diagonal $A$, the ellipse is tilted — and its axes are the eigenvectors of $A$.\
Why $L_1$ gives zeros: the diamond has sharp corners on the axes, so when the loss contours touch it, they tend to hit a corner, where one weight is exactly 0. The circle has no corners, so $L_2$ just shrinks weights.
:::

:::viz norms | Drag $x$; the diamond, circle and square are the points with the same $\|x\|_1$, $\|x\|_2$ and $\|x\|_\infty$. In the ellipse view, set $A$ and the axes follow its eigenvectors.
:::

:::viz l1-corner | Drag the centre of the loss contours. The highlighted point is where the smallest contour touches the diamond or the circle.
:::

### Loss landscapes and contour plots

Suppose a model has params $w=\begin{bmatrix}w_1\\w_2\end{bmatrix}$ and

$$L(w_1,w_2)=w_1^2+w_2^2$$

:::cols
:::col
- The 3D surface looks like a **bowl**.
- Imagine looking at it directly from above.
- If $L=1$, then $w_1^2+w_2^2=1$.
- So you see circles: $L=1$, $L=4$, $L=9$, etc.
- Imagine your model is here (outer circle) and you are training it: it will go down to $L=1$, like a ball rolling down.
:::
:::col
:::fig contours
:::
:::
:::

:::add Added note
The circles have radii $1,2,3$ (since $w_1^2+w_2^2=L$, radius $=\sqrt{L}$). Training keeps rolling inward past $L=9\to4\to1$, all the way toward the minimum $L=0$ at the centre.
:::

:::big Big picture
A contour plot is a bird's-eye view of the loss landscape.
:::

:::viz bowl | Drag to orbit the bowl, switch to the view from above, and let the ball roll from the outer circle.
:::

## 1.1.8 Singular Value Decomposition (SVD)

### What SVD says

:::def SVD says
Any matrix can be understood by **stretching certain special input directions** by certain amounts and **sending them into corresponding output directions**.

$$A=U\,\Sigma\,V^T$$

Read **right to left**: $\quad$ rotate $\;\to\;$ stretch/shrink $\;\to\;$ rotate again.
:::

:::fig svd-strip
:::

**Step 1)** $V^T$ helps rotate the coordinate system so that I am aligned with the directions that $A$ wants to stretch.

:::add Added note
Completing the remaining steps:\
**Step 2)** $\Sigma$ is diagonal: it stretches/shrinks along those aligned directions. Its diagonal entries $\sigma_1\ge\sigma_2\ge\cdots\ge0$ are the *singular values*.\
**Step 3)** $U$ rotates the result into its final orientation in the output space.\
Compared to $A=PDP^{-1}$: SVD works for *any* matrix (even non-square), and $U,V$ are orthogonal (pure rotations/reflections), so $V^{-1}=V^T$. The number of nonzero singular values = rank of $A$.
:::

:::viz svd2 | Drag the column tips to set $A$, then step through $V^T$, $\Sigma$, $U$ (right to left).
:::

### One direction at a time

:::ex E.g. the matrix
$$A=\begin{bmatrix}4&1&2&0\\1&4&0&2\\2&0&4&1\\0&2&1&4\end{bmatrix}\qquad\text{SVD takes this matrix apart into}\qquad A=U\Sigma V^T.$$
:::

Let's take it one direction at a time. The **strongest direction**:

$$v_1^T=\begin{bmatrix}0.5&0.5&0.5&0.5\end{bmatrix},\qquad\sigma_1=7,\qquad u_1=\begin{bmatrix}0.5\\0.5\\0.5\\0.5\end{bmatrix}$$

:::def Reading one direction
- $v_1$ is a special **input direction**.
- $\sigma_1=7$ tells us **how strongly** $A$ stretches that direction.
- $u_1$ is the **output direction** it becomes after the transformation.

$$v_1\;\xrightarrow{\;\;A\;\;}\;\sigma_1u_1\qquad\text{in fact}\qquad\boxed{Av_1=\sigma_1u_1}$$
:::

:::add Added note
Quick check: every row of $A$ sums to 7, so $Av_1=\begin{bmatrix}3.5&3.5&3.5&3.5\end{bmatrix}^T=7\cdot\begin{bmatrix}0.5&0.5&0.5&0.5\end{bmatrix}^T=\sigma_1u_1$. ✓
:::

### Assembling $A$ piece by piece

Now let's start assembling $A_1$ (strongest direction), $A_2$ (second strongest), …

**Keeping one direction:**

$$A_1=\sigma_1\,u_1v_1^T$$

$$u_1v_1^T=\begin{bmatrix}0.5\\0.5\\0.5\\0.5\end{bmatrix}\begin{bmatrix}0.5&0.5&0.5&0.5\end{bmatrix}=\begin{bmatrix}0.25&0.25&0.25&0.25\\0.25&0.25&0.25&0.25\\0.25&0.25&0.25&0.25\\0.25&0.25&0.25&0.25\end{bmatrix}$$

$$\sigma_1u_1v_1^T=7\cdot u_1v_1^T=\begin{bmatrix}1.75&1.75&1.75&1.75\\1.75&1.75&1.75&1.75\\1.75&1.75&1.75&1.75\\1.75&1.75&1.75&1.75\end{bmatrix}$$

That's $A_1$, but $A\neq A_1$: **we are throwing info away**.

:::def We say
If I'm only allowed to keep **ONE** direction/pattern from $A$, keep its **strongest** one.
:::

**And so on.** With $A_2$ we keep the strongest 2, etc.:

$$\begin{aligned}\text{rank 2:}\quad&A_2=\sigma_1u_1v_1^T+\sigma_2u_2v_2^T\\\text{rank 3:}\quad&A_3=\sigma_1u_1v_1^T+\sigma_2u_2v_2^T+\sigma_3u_3v_3^T\end{aligned}$$

Eventually, if we keep **all** the singular components, we reconstruct the original $A$ exactly:

$$A=\sigma_1u_1v_1^T+\sigma_2u_2v_2^T+\sigma_3u_3v_3^T+\sigma_4u_4v_4^T$$

:::def And that is
$$A=U\Sigma V^T\qquad\text{or}\qquad A=\sum_i\sigma_i\,u_iv_i^T$$

Each $\sigma_iu_iv_i^T$ is a rank-1 "layer" of $A$, sorted from strongest to weakest.
:::

:::add Added note
The remaining pieces for this $A$ (it's symmetric with positive eigenvalues, so here $u_i=v_i$ and the singular values equal the eigenvalues):

$$\sigma_2=5,\ v_2=\tfrac12\begin{bmatrix}1\\-1\\1\\-1\end{bmatrix}\qquad\sigma_3=3,\ v_3=\tfrac12\begin{bmatrix}1\\1\\-1\\-1\end{bmatrix}\qquad\sigma_4=1,\ v_4=\tfrac12\begin{bmatrix}1\\-1\\-1\\1\end{bmatrix}$$

So $\Sigma=\operatorname{diag}(7,5,3,1)$, and $A_4=A$. Since the weakest layer has $\sigma_4=1$, dropping it ($A_3$) loses little — this is exactly how SVD does compression and noise reduction (§1.1.2): keep the few largest $\sigma_i$, drop the rest.
:::

:::add Added note
Three more facts that tie this part together:

- $U$ and $V$ are **orthogonal** matrices (§1.1.4): pure rotations. That's why the recipe reads rotate $\to$ stretch $\to$ rotate, and why the formula ends in $V^T$ ($V^{-1}=V^T$).
- Where they come from: $A^TA=V\Sigma^2V^T$ and $AA^T=U\Sigma^2U^T$. So the columns of $V$ are the eigenvectors of $A^TA$, the columns of $U$ are the eigenvectors of $AA^T$, and the $\sigma_i^2$ are the shared eigenvalues.
- **Eckart–Young theorem:** $A_k$ is the *best* rank-$k$ approximation there is — no rank-$k$ matrix comes closer — and the error is exactly the discarded layers:
   $$\|A-A_k\|_F=\sqrt{\sigma_{k+1}^2+\cdots+\sigma_r^2}.$$
   The cost of compression is known before you pay it.
:::

:::viz svd-layers | Move $k$: the layers $\sigma_iu_iv_i^T$ kept so far, their sum $A_k$, and what is still missing, $A-A_k$.
:::

:::ml SVD in ML
$A_k$ is the best rank-$k$ approximation of $A$: the strongest $k$ patterns, with the weak ones (often noise) thrown away. This is the low-rank approximation that PCA and SVD use for compression.
:::

## 1.1.9 Quick Reference

|{stack}
| **Concept** | Formula / definition | Meaning |
|---|---|---|
| **Matrix** | $\begin{bmatrix}a&b\\c&d\end{bmatrix}$ | a linear transformation of the grid |
| **Determinant** | $\lvert\det A\rvert$ | how much the whole grid is stretched |
| $\det A=0$ | basis fails to span the full space | a dimension was lost $\Rightarrow$ not invertible |
| **Rank** | # independent directions | # independent dims that survive |
| **Max rank** | $\min(m,n)$ | never larger than the smallest dimension |
| **Null space** | $\{x:Ax=0\}$ | what the matrix is blind to |
| **Rank–nullity** | rank $+$ nullity $=n$ | survive $+$ disappear $=$ input dims |
| **Eigenvector** | $Av=\lambda v,\ v\neq0$ | direction doesn't change, only scales |
| **Eigenvalues** | $\det(A-\lambda I)=0$ | $A-\lambda I$ must have a non-trivial null space |
| **Diagonalization** | $A=PDP^{-1}$ | $P$ = eigenvectors, $D$ = eigenvalues |
| **Powers** | $A^k=PD^kP^{-1}$ | $D^k$ is easy: power each diagonal entry |
| $L_2$ **norm** | $\sqrt{\sum x_i^2}$ | Euclidean distance; circle; small weights |
| $L_1$ **norm** | $\sum\lvert x_i\rvert$ | Manhattan distance; diamond; sparse weights |
| **Ellipse** | $x^TAx=1$ | some directions more expensive |
| **Regularized loss** | $\min_w[L(w)+\lambda\|w\|]$ | $\lambda$ controls how much we care |
| **SVD** | $A=U\Sigma V^T$ | rotate $\to$ stretch/shrink $\to$ rotate again |
| **One SVD direction** | $Av_i=\sigma_iu_i$ | input dir $v_i$, stretched by $\sigma_i$, becomes $u_i$ |
| **SVD as layers** | $A=\sum_i\sigma_iu_iv_i^T$ | keep top $k$ layers $\Rightarrow A_k$ (rank-$k$ approx.) |
| **Column space** | $\operatorname{Col}(A)=\{A\mathbf{x}\}$ = span of columns | everything $A$ can reach; rank = its dimension |
| **Solving** $A\mathbf{x}=\mathbf{b}$ | solvable $\iff\mathbf{b}\in\operatorname{Col}(A)$ | full rank: one solution; low rank: none or $\infty$ |
| **Transpose** | $(AB)^T=B^TA^T$ | flip across the diagonal; order reverses |
| **Inverse** | $AA^{-1}=I$; exists $\iff\det\neq0$ | undoes $A$; $(AB)^{-1}=B^{-1}A^{-1}$ |
| **Symmetric** | $A^T=A$ | real eigenvalues; $A=Q\Lambda Q^T$ |
| **Orthogonal** | $Q^TQ=I$, so $Q^{-1}=Q^T$ | pure rotation; preserves lengths/angles |
| **Rank-one** | $A=\mathbf{u}\mathbf{v}^T$ | one pattern; what SVD layers are made of |
| $L_\infty$ **/ Frobenius** | $\max_i\lvert x_i\rvert$; $\ \sqrt{\sum a_{ij}^2}$ | biggest coordinate; $L_2$ of a flattened matrix |
| **Eckart–Young** | $\|A-A_k\|_F=\sqrt{\sum_{i>k}\sigma_i^2}$ | $A_k$ is the best rank-$k$ approximation |
