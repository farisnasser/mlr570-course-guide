# Visual plan — MLR570 Course Guide

Source: `reference/MLR570_Course_Notes.pdf` (119 pp., 3 chapters / 9 parts). Read in full before planning.

Rules every visual follows
- Adds no information. Labels, captions and readouts use only the notes' own terms, symbols and numbers; one caption line says what to drag.
- Runs the real math: real matrices (det, inverse, eigen, SVD computed live), real distributions (exact pdf/cdf, real sampling), real solvers (SVM QP, OLS/WLS/ridge closed forms, kernel ridge, k-NN).
- Where the notes have a worked example, the visual opens on exactly that example and shows exactly those numbers
  (each one recomputed in `scripts/check/<part>.py`).
- The PDF's own figures stay in the notes as exact vector extracts (`scripts/extract_figures.py`); interactive visuals sit
  next to the passage they explain, in a frame marked "Interactive", so they never read as part of the notes.
- Colour language (shared with the Survey Atlas): first column / e1 green `#5fe0b0`, second column / e2 coral `#ff7a59`,
  span / column space blue `#7cc4ff`, null space / collapse red `#ff5d73`, eigen directions amber `#ffc857`,
  free vector violet `#b28cff`, ink `#e8ecf8`.

Inspiration (ideas only, no code or assets): 3Blue1Brown *Essence of Linear Algebra* (grid transformations, det as
area, eigenvectors staying on their span, change of basis), *Essence of Calculus* (secant→tangent, chain rule as
nudges), Seeing Theory (probability distributions, sampling, Bayes), setosa.io *Explained Visually* (PCA, conditional
probability, least squares), Immersive Math (rank/null space), distill.pub (interaction honesty), R2D3 (decision
boundaries), Mathigon (step-by-step reveals).

---

## Part 1.1 Linear Algebra (pp. 5–22) — highest priority (Faris: "especially the maths at the beginning")

| # | Section | Concept | Interaction | Why it helps |
|---|---|---|---|---|
| 1 | 1.1.1 | Matrix = linear transformation; det = area stretch; det = 0 collapse | Drag the column tips $Ae_1=(a,c)$, $Ae_2=(b,d)$; the whole grid and unit square follow; a $t$ slider morphs $I\to A$. Readout: $A$, $\det A=ad-bc$, orientation flip, invertible or not. Presets: the notes' $\begin{bmatrix}4&7\\2&6\end{bmatrix}$ (det 10), $\begin{bmatrix}2&4\\1&2\end{bmatrix}$ (det 0), $Q$ (45° rotation), $\begin{bmatrix}2&1\\1&2\end{bmatrix}$. When det = 0 the plane collapses onto Col$(A)$ and a red null-space line shows the many inputs landing on one output. | "Area scaled by $\lvert\det A\rvert$" and "a dimension was lost" become something you watch happen. |
| 2 | 1.1.1 | $AB$ = apply $B$ first; order matters | Two stages side by side: "stretching then rotating" vs "rotating then stretching", with stretch and angle sliders; both final grids drawn over each other. | $AB\neq BA$ is seen, not memorised. |
| 3 | 1.1.2 | Solving $A\mathbf x=\mathbf b$: three cases; $\mathbf b\in$ Col$(A)$ | Row picture (the two lines) next to column picture (Col$(A)$ and $\mathbf b$). Case buttons load the notes' three systems; drag $\mathbf b$. Unique: $\mathbf x=(1,2)$; rank-deficient: lines parallel until $\mathbf b$ lands on the column line, then they coincide. | Ties the elimination examples to the column-space statement in the boxed rule. |
| 4 | 1.1.3 | Rank–nullity, null space as blind spot | three.js: input space $\mathbb R^3$ with the null-space plane of the notes' $A=\begin{bmatrix}1&2&-1\\2&4&-2\end{bmatrix}$ (spanned by $(1,0,1)^T$, $(0,1,2)^T$); output $\mathbb R^2$ with Col$(A)$ = a line. Drag $x$; move it inside the null plane and $Ax$ does not move ($A(x+z)=Ax$). Readout rank 1 + nullity 2 = 3. | The ML box's "$x$ and $x+z$ are identical to the model" made literal. |
| 5 | 1.1.5 | Eigenvectors keep their direction | A fan of vectors under $t\!:\,I\to A$; most leave their line, the eigen-directions stay on their span (amber) and stretch by $\lambda$. Drag one vector $v$, see $Av$; side chart of $\det(A-\lambda I)$ vs $\lambda$ with its roots. Default the notes' $A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$: $\lambda=3,1$, $v=(1,1),(1,-1)$. | "Eigenvector directions do not change" and "why $\det(A-\lambda I)=0$" in one picture. |
| 6 | 1.1.6 | $A=PDP^{-1}$ read right to left; $A^k=PD^kP^{-1}$ | Step player $P^{-1}\to D\to P$ morphing the grid and the eigen-grid, ending exactly on $A$; power slider $k$ shows $D^k$ and $A^k$ (notes' $P$, $D$). | Makes the 3-step reading of the factorisation visual. |
| 7 | 1.1.7 | One vector, three sizes; unit balls | Drag $x$ (default $(3,-4)$): the $L_1$ diamond, $L_2$ circle and $L_\infty$ square through $x$ with $\lVert x\rVert_1=7$, $\lVert x\rVert_2=5$, $\lVert x\rVert_\infty=4$; ellipse $x^TAx=1$ mode with draggable symmetric $A$ (default diag(1,4); tilted when off-diagonal, axes = eigenvectors). | The shapes-of-equal-norm figure becomes explorable; the added note about tilted ellipses is shown. |
| 8 | 1.1.7 | Why $L_1$ gives zeros, $L_2$ shrinks | Elliptic loss contours around a draggable unconstrained minimum; exact constrained minimiser on the $L_1$ diamond vs the $L_2$ circle; readout shows when a weight is exactly 0. | The added note "they tend to hit a corner" checked live with a real solver. |
| 9 | 1.1.7 | Loss landscape ↔ contour plot | three.js bowl $L=w_1^2+w_2^2$ with contours $L=1,4,9$; camera glides to the bird's-eye view; a ball rolls down from "model starts here". | "A contour plot is a bird's-eye view of the loss landscape." |
| 10 | 1.1.8 | SVD = rotate → stretch → rotate | Drag the columns of a $2\times2$ $A$; a stage slider runs $V^T$, $\Sigma$, $U$ on the unit circle with $v_1,v_2$; readout $\sigma_1,\sigma_2$, $Av_i=\sigma_iu_i$. | The notes' four-panel strip as one continuous motion for any matrix. |
| 11 | 1.1.8 | SVD as layers; Eckart–Young | The notes' $4\times4$ $A$: slider $k=1..4$ shows the layers $\sigma_iu_iv_i^T$, $A_k$ and $A-A_k$ as heat-tables with entries ($A_1$ = all 1.75), and $\lVert A-A_k\rVert_F=\sqrt{\sigma_{k+1}^2+\dots}$ ($\sqrt{35},\sqrt{10},1,0$). | The worked SVD example and the added Eckart–Young note on the same numbers. |

## Part 1.2 Probability & Statistics (pp. 23–37)
1. **PMF ↔ CDF (1.2.1)** — the notes' PMF $p(0)=\tfrac14,p(1)=\tfrac12,p(2)=\tfrac14$; drag bar heights (kept summing to 1), CDF staircase updates, hover gives $F_X(1)=\tfrac34$.
2. **PDF area = probability (1.2.1)** — Uniform / Normal / Exponential; drag $a,b$: shaded area $P(a\le X\le b)$ next to the CDF gap $F(b)-F(a)$; squeeze $a\to b$ to see $P(X=x)=0$; 68–95–99.7 toggle.
3. **Mean as balance point, variance as squared deviations (1.2.2)** — sets $A=\{9,10,11\}$, $B=\{0,10,20\}$ draggable on a line; signed deviations cancel ($E[X-\mu]=0$), squares give Var $=2/3$ vs $200/3$, $\sigma\approx0.82, 8.16$.
4. **Sampling: $\bar X_n$, $S_n^2$ (1.2.2)** — repeated i.i.d. draws; histogram of $\bar X_n$ narrows as $\sigma^2/n$; dividing by $n$ vs $n-1$ averaged over many repeats; the notes' sample $(3,5,2,4)$: $\bar X=3.5$, $S^2=5/3$.
5. **Joint → marginal → conditional on the 80-student table (1.2.3–1.2.4)** — click cells/rows/columns: $36/80$, $50/80$, $9/45$; conditioning renormalises a row into a new distribution summing to 1.
6. **Continuous joint density sliced (1.2.3–1.2.5)** — three.js bivariate normal ($\rho$ slider); slice at $y=y_0$, normalised slice = $f_{X\mid Y}$; marginal as integral; at $\rho=0$ all slices have the same shape (independence).
7. **Bayes diagnostic test (1.2.4)** — area diagram of the population; prevalence, sensitivity, false-positive sliders; $P(+)=0.059$, $P(D\mid+)\approx0.161$.
8. **Independence table test (1.2.5)** — the notes' $2\times2$ table vs product of marginals; drag a cell to break independence.
9. **Covariance as signed rectangles (1.2.7)** — scatter with $(x-\mu_X)(y-\mu_Y)$ rectangles coloured by sign; $\rho$ readout; the table example $E[XY]=2.6$, Cov $=0.05$, $\rho\approx0.22$.
10. **Law of total expectation (1.2.8)** — weighted subgroup averages: $0.75(80)+0.25(60)=75$.

## Part 1.3 Calculus & Optimization (pp. 38–49)
1. **Secant → tangent (1.3.1)** — $f(x)=x^2$, drag $x$ and $h$; slope → $f'(x)=2x$ (6 at $x=3$); linearisation $\Delta f\approx6(0.01)=0.06$.
2. **Partial derivatives as slices; gradient ⟂ contours (1.3.2)** — three.js surface $f=3x^2y$ with $x$- and $y$-slices, slopes $6xy$, $3x^2$; contour map with $\nabla f$.
3. **Jacobian = local linear map (1.3.3)** — the notes' $(f,g)=(3x^2y,\,2x+y^8)$: a small grid around a point and its image under $(f,g)$ vs under $J$.
4. **Chain rule & total derivative graph (1.3.4)** — $y=\sin(x^2)$ and $y=x+x^2$: values forward, local derivatives backward, paths summed; the simple neuron $z=w^Tx+b$, $a=\sigma(z)$.
5. **Gradient descent and the step size (1.3.6)** — $L=w^Tw$, $w_{t+1}=(1-2\eta)w_t$: $\eta$ slider shows converge / oscillate / diverge ($\eta>1$); 2D contour version.
6. **Convexity (1.3.6)** — chord above graph with $\lambda$ slider; first-order test tangent; Hessian eigenvalue signs → minimum / maximum / saddle surfaces (three.js, Atlas curvature colours).

## Part 2.1 Data Preprocessing (pp. 50–62)
1. **Scale distorts geometry (2.1.1, 2.1.4)** — two features $\sim10^4$ vs $\sim10$; distance $\approx1000.01$; nearest neighbour changes after min–max / z-score.
2. **Mean vs median, rolling window (2.1.2)** — drag the outlier in $\{2,3,3,4,100\}$ (22.4 vs 3); noisy series with window $w$: lag vs noise.
3. **Sampling & imbalance (2.1.3)** — random vs stratified split proportion spread; 40 vs 4 over/undersampling.
4. **Min–max outlier squeeze (2.1.4)** — ages $\{20,30,40,60\}$ (+100): $30\mapsto0.25\to0.125$.
5. **One-hot geometry (2.1.5)** — label encoding on a line vs one-hot basis vectors in 3D, all pairs $\sqrt2$ apart.
6. **Binning (2.1.6)** — equal-width vs equal-frequency vs k-means bins on one skewed sample (counts 9,3,1,1 vs ≈4,4,3,3).
7. **PCA (2.1.8)** — rotate a candidate axis through a 2D cloud; projected variance peaks at the top eigenvector of $\Sigma$; explained-variance bars (70/20/10); 3D cloud → 2D.

## Part 2.2 Similarity Measures (pp. 63–68)
1. **Pearson on draggable scatter (2.2.1)** — linear $B=2A$ ($\rho=1$) and $B=A^2$ ($\rho=0$).
2. **Euclidean vs Manhattan (2.2.2)** — drag $A,B$: straight line vs every staircase path (same $L_1$ length); $\sqrt{27}$ vs 9.
3. **Curse of dimensionality (2.2.3)** — simulation of $(d_{\max}-d_{\min})/d_{\min}$ and distance histograms as $d$ grows.
4. **Cosine vs Euclidean (2.2.4)** — angle-only similarity; $A=[1,2,3]$, $C=2A$: cos 1, distance $\sqrt{14}$.
5. **Jaccard (2.2.5)** — set Venn and binary vectors; $2/7$, $0.5$; shared absences ignored.

## Part 2.3 Model Evaluation (pp. 69–76)
1. **Under/overfitting (2.3.2)** — polynomial degree slider; training vs validation error curves.
2. **K-fold CV (2.3.3)** — folds rotate, per-fold error, average.
3. **Threshold on a confusion matrix (2.3.4–2.3.7)** — two score distributions, slide $t$: TP/FP/TN/FN, precision, recall, F1, F$_\beta$ ($\beta$ slider); the lecture rows reproduced.
4. **Harmonic vs geometric vs arithmetic mean (2.3.5)** — $P=0.1,R=0.9$: 0.18 ≤ 0.3 ≤ 0.5.
5. **MSE vs MAE and the outlier (2.3.8)** — drag one target: 0.375/0.5 → 1693.1/18.8; best constant = mean vs median.

## Part 3.1 Hard-Margin SVM (pp. 77–90)
1. **Line geometry (3.1.2–3.1.3)** — drag $w$, $b$: normal vector, signed score field, perpendicular foot $z$, distance $\lvert w^Tx_0+b\rvert/\lVert w\rVert$ unchanged by scaling (the $x+y=4$, $(0,0)$ → $2\sqrt2$ example).
2. **Maximum margin on the lecture's ten points (3.1.4, 3.1.11–3.1.12)** — real hard-margin QP; $w^*=(8/29,13/29)$, $b^*=-13/29$, margin 3.80, three support vectors; drag points and re-solve; non-support points move freely without changing the separator; compare with $x_1+x_2=0.5$.
3. **Dual = best lower bound (3.1.9)** — toy problem: $L(x,\alpha)$ in $x$ for a chosen $\alpha$, and $g(\alpha)=\alpha-\alpha^2/4$ rising to $p^*=1$ at $\alpha=2$.
4. **Dual weights (3.1.10–3.1.11)** — $\alpha_i$ per point: nonzero only on support vectors.

## Part 3.2 OLS, WLS & Ridge (pp. 91–101)
1. **OLS = orthogonal projection (3.2.3–3.2.4)** — three.js $\mathbb R^3$: Col$(X)$ = span$\{\mathbf 1,x\}$ for the notes' $X$, $y=(1,2,2)$; drag a candidate $X\beta$ in the plane; the foot $\hat y$ gives $\hat\beta=(0.6667,0.5)$, $e\perp$ plane; linked data-space plot.
2. **Heteroscedasticity and WLS (3.2.5–3.2.6)** — funnel data; OLS vs WLS with $w_i\propto1/\sigma_i^2$.
3. **Multicollinearity & ridge (3.2.7–3.2.9)** — the notes' $4\times3$ example: OLS $(-1.225,1.5,2.5)$ → ridge $\lambda=1$ $(0.835,0.801,0.811)$; coefficient paths in $\lambda$; eigenvalues of $X^TX+\lambda I$ lifted; noise re-draws make OLS swing while ridge stays put.

## Part 3.3 Kernels & k-NN (pp. 102–119)
1. **Feature lift (3.3.1)** — three.js ring data lifted by $\varphi(x)=(x_1,x_2,x_1^2+x_2^2)$; plane at 2.1 ↔ circle radius 1.45.
2. **Polynomial kernel = dot product (3.3.4)** — drag $x,x'$: $(1+x^Tx')^2$ vs $\varphi(x)^T\varphi(x')$ term by term (36 at $(1,2),(3,1)$); feature-count explosion $\binom{n+d}{d}$.
3. **Kernel perceptron (3.3.5)** — step through training; $\alpha_j$ counters per point; boundary with polynomial / RBF kernel.
4. **RBF $\gamma$: local vs global (3.3.6–3.3.9)** — bumps and decision boundary as $\gamma$ changes; RBF vs Laplacian profile.
5. **Kernel validity (3.3.10)** — Gram-matrix eigenvalues for chosen points and kernel (PSD check).
6. **Kernel ridge regression (3.3.11)** — 1D data: OLS line vs RBF kernel regression; $\alpha_iK(x_i,x)$ bumps summing to the prediction; $\gamma$, $\lambda$ sliders.
7. **k-NN (3.3.12–3.3.15)** — query star with $k=5$ (3 red, 2 blue); decision regions vs $k$; weighted vote flip (distances 1, 2, 4); k-NN regression from $k=1$ to large $k$; validation MSE vs $k$; nearest/farthest/average distance vs $d$ simulation.
