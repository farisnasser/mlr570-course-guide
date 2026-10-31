:::plain
**The flow:** introduction → Pearson correlation → Euclidean distance → Manhattan distance → curse of dimensionality → cosine similarity → Jaccard similarity → comparative summary.
:::

Similarity and distance measures give us a numerical way to describe how alike or different two observations are. The key idea is that "similar" does not always mean the same thing: sometimes we care about **linear co-movement**, sometimes **geometric distance**, sometimes **direction**, and sometimes **set overlap**.

:::ml Why this matters in ML
Many algorithms make decisions by comparing observations. $k$-NN and K-means rely on distance, feature analysis often uses correlation, and text or embedding systems commonly use cosine or Jaccard similarity. The measure we choose determines the **geometry the algorithm "sees."**
:::

## 2.2.1 Pearson Correlation

:::def Pearson correlation
Measures the strength and direction of the **linear** relationship between two variables (§1.2.7):

$$\rho(A,B)=\frac{\operatorname{Cov}(A,B)}{\sigma_A\,\sigma_B}$$
:::

Its value lies between $-1$ and 1:

| | |
|---|---|
| $\rho=1$ | perfect positive linear relationship |
| $\rho=0$ | no linear relationship detected |
| $\rho=-1$ | perfect negative linear relationship |

:::ex Linear example
Let $A=[1,2,3,4,5]$ and $B=[2,4,6,8,10]$. Since $B=2A$, the variables lie exactly on a straight line and $\rho=1$.
:::

:::ex Nonlinear example
:::cols
:::col
Let $B=A^2$ with values of $A$ symmetric around zero. $B$ is **completely determined** by $A$, but Pearson correlation can still be 0 because the increasing and decreasing linear trends cancel.\
E.g. $A=[-2,-1,0,1,2]$, $B=[4,1,0,1,4]$. Here $E[A]=0$, so

$$\operatorname{Cov}(A,B)=E[AB]-E[A]E[B]=\tfrac{-8-1+0+1+8}{5}-0=0$$

$\Rightarrow\ \rho=0.$
:::
:::col
:::fig parabola
:::
:::
:::
:::

:::add Added clarification
**Important:** $\rho=0$ does **not** mean "no relationship." It means Pearson detected no *linear* relationship. Nonlinear dependence may still be very strong.
:::

**Applications:** feature selection, detecting strongly correlated variables, and analyzing linear relationships in regression.\
**Limitation:** Pearson misses nonlinear relationships and is sensitive to outliers.

:::viz pearson | Drag the points. Start from the notes' two examples; then pull one point far away to see how much a single outlier moves $\rho$.
:::

## 2.2.2 Euclidean and Manhattan Distance

Both are **distance** measures: smaller distance means the points are closer. The difference is the geometry used to measure that closeness.

### Euclidean distance

:::def Euclidean distance
The ordinary straight-line distance between two points. It is the $L_2$ distance from Chapter 1 (§1.1.7).

$$d(A,B)=\sqrt{\sum_{i=1}^n(A_i-B_i)^2}$$
:::

:::ex Numerical example
$A=[1,2,3]$, $B=[4,5,6]$. Then

$$d(A,B)=\sqrt{(1-4)^2+(2-5)^2+(3-6)^2}=\sqrt{27}\approx5.196.$$
:::

Typical applications include $k$-NN classification and K-means clustering. Its main weaknesses are **sensitivity to feature scaling** (§2.1.4) and reduced discriminative power in very high dimensions.

### Manhattan distance

:::def Manhattan distance
The sum of the absolute coordinate-wise differences. It is the $L_1$ distance from Chapter 1 (§1.1.7).

$$d(A,B)=\sum_{i=1}^n|A_i-B_i|$$
:::

:::ex Same points
$A=[1,2,3]$, $B=[4,5,6]$. Then

$$d(A,B)=|1-4|+|2-5|+|3-6|=9.$$
:::

:::fig paths
:::

:::ml Geometry
Euclidean = straight-line path. Manhattan = move only along coordinate axes, like navigating a city grid.
:::

:::add Intuition
Because Euclidean distance **squares** each coordinate difference, one unusually large difference can dominate strongly. Manhattan grows only **linearly** with that difference, so it is often more robust to outliers.\
E.g. differences $[1,1,10]$: Euclidean $=\sqrt{1+1+100}\approx10.1$ (almost all from the 10), Manhattan $=1+1+10=12$.
:::

:::viz distances | Drag $A$ and $B$. The straight segment is the Euclidean distance; every staircase along the grid has the same Manhattan length.
:::

## 2.2.3 The Curse of Dimensionality

:::def Curse of dimensionality
As the number of dimensions grows, distance-based comparisons can become **less informative**. Euclidean distances between different points tend to become more similar to one another, making "nearest" and "farther" harder to distinguish.
:::

In low dimensions, one neighbor can clearly be much closer than another. In very high dimensions, every pair differs across many coordinates, so distances accumulate many contributions.

:::add Added intuition
Mental picture: as dimensionality grows, the **gap between the nearest and farthest points can shrink relative to their overall distance**. This is why methods such as $k$-NN and clustering can struggle in high-dimensional spaces.

$$\frac{d_{\max}-d_{\min}}{d_{\min}}\;\longrightarrow\;0\qquad\text{as }d\text{ grows}$$

If every point is roughly the same distance away, "nearest neighbor" stops meaning much.
:::

The lecture notes emphasize that Manhattan distance can sometimes be more stable than Euclidean distance in high-dimensional settings because it treats coordinate differences linearly rather than squaring them.

:::ml What we do about it
This is one more reason for dimensionality reduction and feature selection (§2.1.7–2.1.9): fewer, more informative dimensions keep distances meaningful.
:::

:::viz curse | Move the dimension $d$: 1000 uniform random points, one random query. The strip shows every distance divided by the average; the curve tracks $(d_{\max}-d_{\min})/d_{\min}$.
:::

## 2.2.4 Cosine Similarity

:::def Cosine similarity
Measures the cosine of the **angle** between two vectors. It asks whether they point in the same direction rather than how far apart their endpoints are.

$$\cos(A,B)=\frac{A\cdot B}{\|A\|\,\|B\|}$$
:::

:::cols
:::col
| | |
|---|---|
| $\cos=1$ | same direction |
| $\cos=0$ | perpendicular directions |
| $\cos=-1$ | opposite directions |
:::
:::col
:::fig angle
:::
:::
:::

:::ex Perfect alignment
$A=[1,2,3]$ and $C=[2,4,6]=2A$. $\ C$ is twice as large, but it points in exactly the same direction:

$$\cos(A,C)=\frac{2+8+18}{\sqrt{14}\cdot\sqrt{56}}=\frac{28}{\sqrt{14}\cdot2\sqrt{14}}=\frac{28}{28}=1.$$
:::

:::ml Key distinction
**Euclidean** asks: "How far apart are these vectors?"\
**Cosine** asks: "How aligned are these vectors?"\
Two vectors can be far apart in magnitude and still have cosine similarity 1.
:::

**Applications:** document similarity, recommender systems, and high-dimensional embeddings such as word or image embeddings. Cosine is useful when direction/pattern matters more than magnitude.

**Link to Pearson:** if you first center both vectors (subtract their means), cosine similarity becomes exactly Pearson correlation. So Pearson is cosine on mean-centered data.

:::viz cosine | Drag the tips of $A$ and $B$, or stretch $B$ with the slider: the cosine only changes when the angle does, while the Euclidean distance changes with length too.
:::

## 2.2.5 Jaccard Similarity

:::def Jaccard similarity
Measures how much two **sets** overlap.

$$J(A,B)=\frac{|A\cap B|}{|A\cup B|}$$
:::

:::ex Set example
:::cols
:::col
$A=\{1,2,3,4,5\}$, $B=\{4,5,6,7\}$.\
Intersection $=\{4,5\}$ has size 2; union $=\{1,2,3,4,5,6,7\}$ has size 7. Therefore

$$J(A,B)=\frac27\approx0.286.$$
:::
:::col
:::fig venn
:::
:::
:::
:::

:::ex Binary-vector example
$A=[1,0,1,0,1]$, $B=[1,1,0,0,1]$. Treat the 1-entries as set membership. Two positions are shared (positions 1 and 5) and four positions occur in the union (1, 2, 3, 5), so

$$J=\frac24=0.5.$$

Position 4 is 0 in both, so it counts in neither: Jaccard ignores shared absences.
:::

**Interpretation:** $J=1$ means identical sets; $J=0$ means no overlap. Jaccard is intuitive for documents, categorical data, and binary feature vectors, but it **ignores frequency and magnitude**.

:::viz jaccard | Click the cells to switch entries of $A$ and $B$ between 0 and 1. Positions that are 0 in both never count.
:::

## 2.2.6 Comparative Summary

The most important idea is that each measure answers a **different question**. "Similarity" only makes sense after deciding what aspect of the data should count as similar.

|{stack}
| **Measure** | What it asks | Best suited for | Main limitation |
|---|---|---|---|
| **Pearson** | Do the variables move together linearly? | feature relationships, regression | misses nonlinear dependence; outlier-sensitive |
| **Euclidean** | How far apart in a straight line? | low-dimensional numeric data, $k$-NN, K-means | scale-sensitive; weaker in high dimensions |
| **Manhattan** | How far apart axis-by-axis? | grid-like or high-dimensional data | less intuitive geometrically |
| **Cosine** | Do the vectors point in the same direction? | text, embeddings, recommender systems | ignores magnitude differences |
| **Jaccard** | How much set membership overlaps? | sets, binary/categorical data | ignores frequency/magnitude |

:::model Mental map
| | | |
|---|---|---|
| Relationship between variables | $\to$ | **Pearson** |
| Numerical/geometric distance | $\to$ | **Euclidean** or **Manhattan** |
| Direction or pattern, not size | $\to$ | **Cosine** |
| Shared membership or overlap | $\to$ | **Jaccard** |
:::

:::ml Why the choice matters
$A=[1,2,3]$ and $B=[2,4,6]$ have cosine similarity 1 because they point in the same direction, yet their Euclidean distance is $\sqrt{1+4+9}=\sqrt{14}\approx3.74$, not zero, because their magnitudes differ. The same pair can therefore be **similar under one definition and separated under another**.
:::

## 2.2.7 Quick Reference

|{stack}
| **Concept** | Formula / definition | Meaning |
|---|---|---|
| **Pearson correlation** | $\rho=\operatorname{Cov}(A,B)/(\sigma_A\sigma_B)$ | linear association; $-1$ to 1 |
| $\rho=0$ | no linear correlation | does not rule out nonlinear dependence |
| **Euclidean** | $\sqrt{\sum(A_i-B_i)^2}$ | straight-line / $L_2$ distance |
| **Manhattan** | $\sum\lvert A_i-B_i\rvert$ | axis-wise / $L_1$ distance |
| **Curse of dimensionality** | distance contrast decreases as $d$ grows | nearest/farthest become less distinguishable |
| **Cosine** | $\dfrac{A\cdot B}{\|A\|\,\|B\|}$ | orientation; ignores overall scale |
| **Jaccard** | $\lvert A\cap B\rvert/\lvert A\cup B\rvert$ | set/binary overlap |

> **Source note.** The core content and ordering in these notes follow the MLR570 Chapter 2 revised lecture notes. Amber boxes mark intuitive clarifications developed during study so they remain distinguishable from the lecture-source material.
