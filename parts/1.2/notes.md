:::plain
**The flow:** Random variable → PMF / PDF → CDF → Moments (mean, variance, std) → Two random variables → Joint → Marginal → Conditional → Independence → Expectation of two variables → Covariance → Correlation → Conditional expectation → Law of total expectation.
:::

:::fig rv-tree
:::

## 1.2.1 A Single Random Variable

### Sample space and random variables

The **sample space** $\Omega$ is the set of all possible outcomes of a random experiment: tossing two coins gives $\Omega=\{HH,HT,TH,TT\}$; measuring a temperature gives $\Omega=\mathbb{R}$.

A **random variable** $X$ assigns a number to each outcome, so we can work with numbers instead of raw outcomes. Notation: capital $X$ is the random variable, lowercase $x$ is a particular value it took.

A random variable $X$ is either **discrete** (it takes countable values, like $0,1,2$) or **continuous** (it takes any value in a range). Each one has its own way of describing probability.

### Discrete: PMF

:::def PMF
$$p_X(x)=P(X=x)\qquad\qquad\sum_x p_X(x)=1$$

The PMF gives the probability **at one value**.
:::

### Discrete: CDF

:::def CDF
$$F_X(x)=P(X\le x)$$

So it accumulates probability as $x$ increases.
:::

:::ex Example
:::cols figleft
:::col
:::fig pmf-table
:::
:::
:::col
**CDF at 1:**

$$P(X\le1)=P(X=0)+P(X=1)$$

$$\text{so}\quad P(X\le1)=\tfrac14+\tfrac12=\tfrac34$$
:::
:::

:::fig pmf-cdf-plots
:::
:::

:::viz pmf-cdf | Drag the tops of the bars (they always sum to 1) and move $x$ along the axis; the CDF adds up everything at or left of $x$.
:::

### Continuous: PDF

:::def PDF
$$P(a\le X\le b)=\int_a^b f_X(x)\,dx\qquad\qquad\int_{-\infty}^{\infty}f_X(x)\,dx=1$$
:::

:::def Rule
For a continuous random variable, at a specific point:

$$P(X=x)=0$$

So $f_X(x)$ is **not** interpreted as a probability. Probability comes from an **area**.
:::

:::fig pdf-cdf-curves
:::

### Continuous: CDF

:::def CDF
$$F_X(x)=P(X\le x)=\int_{-\infty}^{x}f_X(t)\,dt$$

It is the area accumulated from far left **up to** $x$.
:::

### PDF $\leftrightarrow$ CDF

Since the CDF is the integral of the PDF, going back is just differentiating:

$$f_X(x)=\frac{d}{dx}F_X(x)$$

:::ml
$$\text{PDF}\xrightarrow{\ \text{integrate}\ }\text{CDF}$$

$$\text{CDF}\xrightarrow{\ \text{differentiate}\ }\text{PDF}$$
:::

:::viz pdf-area | Drag $a$ and $b$: the shaded area under $f_X$ is $P(a\le X\le b)$, the same number as the rise of $F_X$ from $a$ to $b$. Push $a$ onto $b$ and the probability goes to 0.
:::

## 1.2.2 Moments

Moments summarize a distribution with a few numbers: where it is centred, and how spread out it is.

### 1) Mean $E[X]$

:::def Mean
$$E[X]=\sum_x x\,p_X(x)\quad(\text{discrete})\qquad\qquad E[X]=\int_{-\infty}^{\infty}x\,f_X(x)\,dx\quad(\text{continuous})$$

Each term is **value $\times$ how likely it occurs**. We write $E[X]=\mu$.
:::

The mean alone is not enough. Consider

$$A=\{9,\ 10,\ 11\}\qquad\qquad B=\{0,\ 10,\ 20\}$$

Both have mean $=10$, but they are spread out differently.

:::fig ab-dots
:::

To find deviations we can do (for $A$):

$$9-10=-1,\qquad10-10=0,\qquad11-10=1\qquad\Rightarrow\qquad\frac{-1+0+1}{3}=0$$

The positives and negatives always cancel:

$$E[X-\mu]=0\quad\text{always}$$

$\to$ So the solution is to **square the deviations**.

### 2) Variance

:::def Variance
$$\operatorname{Var}(X)=E\big[(X-\mu)^2\big]$$

Average squared distance from the mean.
:::

**Shortcut formula.** Expand the square: $(X-\mu)^2=X^2-2\mu X+\mu^2$

$$\begin{aligned}\operatorname{Var}(X)&=E\big[X^2-2\mu X+\mu^2\big]\\&=E[X^2]-2\mu\underbrace{E[X]}_{=\mu}+\mu^2\\&=E[X^2]-\mu^2\end{aligned}$$

$$\boxed{\operatorname{Var}(X)=E[X^2]-\big(E[X]\big)^2}$$

:::ex Back to $A$ and $B$
$$\operatorname{Var}(A)=\frac{(-1)^2+0^2+1^2}{3}=\frac23\qquad\qquad\operatorname{Var}(B)=\frac{(-10)^2+0^2+10^2}{3}=\frac{200}{3}$$

Same mean, but now the spread shows up: $B$ is way more spread out.
:::

### 3) Standard deviation

:::def Std. deviation
$$\sigma=\sqrt{\operatorname{Var}(X)}$$

Since variance gives values in their units **squared**, the square root brings us back to the original units.
:::

For the example: $\sigma_A=\sqrt{2/3}\approx0.82$ and $\sigma_B=\sqrt{200/3}\approx8.16$.

:::viz moments | Drag the points of $A$ and $B$. The bars are the deviations from the mean (they always cancel); the squares are the squared deviations, whose average is the variance.
:::

:::ml Moments in ML
- Mean and std are what we use to **standardize features**: $z=\dfrac{x-\mu}{\sigma}$, so every feature is centred at 0 with spread 1.
- The MSE loss is a variance-style quantity: the average squared distance between predicted and actual.
:::

### Common distributions

The named distributions are just ready-made PMFs/PDFs for situations that keep recurring.

**Discrete:**

|{stack}
| **Distribution** | $P(X=k)$ | $E[X],\ \operatorname{Var}(X)$ | Models |
|---|---|---|---|
| **Bernoulli($p$)** | $p$ if $k$=1; $\ 1-p$ if $k$=0 | $p,\ \ p(1-p)$ | one yes/no outcome: a coin flip, a spam indicator |
| **Binomial($n,p$)** | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np,\ \ np(1-p)$ | # successes in $n$ independent trials: defects in a batch |
| **Poisson($\lambda$)** | $\dfrac{\lambda^ke^{-\lambda}}{k!}$ | $\lambda,\ \ \lambda$ | # events per window: arrivals per hour, packets per second |
| **Multinomial** | generalizes binomial | — | counts over $K$ categories |

Poisson has mean = variance = $\lambda$, which is a useful check: count data spread much more widely than its mean is not Poisson.

:::ex Example
$X\sim$ Bernoulli$(0.3)$: $\ E[X]=0.3$ and $\operatorname{Var}(X)=E[X^2]-(E[X])^2=0.3-0.09=0.21$.
:::

**Continuous:**

|{stack}
| **Distribution** | Density | Models |
|---|---|---|
| **Uniform($a,b$)** | constant $\tfrac{1}{b-a}$ on $[a,b]$ | every value in the range equally likely |
| **Normal($\mu,\sigma^2$)** | bell curve centred at $\mu$ | noise, measurement errors, heights |
| **Exponential($\lambda$)** | $\lambda e^{-\lambda x},\ x\ge0$ | waiting time between Poisson events; memoryless: $P(X>s{+}t\mid X>s)=P(X>t)$ |
| **Multivariate Normal** | normal in $d$ dimensions | default model for continuous feature vectors |

:::def The 68–95–99.7 rule
For a Normal$(\mu,\sigma^2)$: about 68% of the probability lies within $\mu\pm\sigma$, about 95% within $\mu\pm2\sigma$, and about 99.7% within $\mu\pm3\sigma$.
:::

:::fig normal-rule
:::

### Sample mean and sample variance

Everything above assumes we *know* the distribution. In practice we have $n$ i.i.d. samples $X_1,\dots,X_n$ from an unknown distribution with true mean $\mu$ and variance $\sigma^2$, and we **estimate**:

$$\boxed{\bar X_n=\frac1n\sum_{i=1}^nX_i}\qquad\qquad\boxed{S_n^2=\frac{1}{n-1}\sum_{i=1}^n\big(X_i-\bar X_n\big)^2}$$

- $E[\bar X_n]=\mu$: the sample mean is an unbiased estimate of the true mean.
- $\operatorname{Var}(\bar X_n)=\sigma^2/n$: averaging $n$ samples shrinks the noise by a factor of $n$. More data $\to$ steadier estimate.
- The $n-1$ (not $n$) makes $S_n^2$ unbiased: the samples sit slightly closer to their own mean $\bar X_n$ than to the true $\mu$, and dividing by $n-1$ corrects exactly for that.

:::ex Example
Sample $(3,5,2,4)$, $n=4$: $\ \bar X=\dfrac{14}{4}=3.5$ and

$$S^2=\frac{(-0.5)^2+(1.5)^2+(-1.5)^2+(0.5)^2}{3}=\frac53\approx1.67.$$
:::

:::viz sampling | Draw samples of size $n$ from the PMF example above ($\mu=1$, $\sigma^2=0.5$). The histogram collects $\bar X_n$; the running averages compare dividing by $n-1$ with dividing by $n$.
:::

## 1.2.3 Two Random Variables: Joint Distributions

Now every outcome gives us **two** values $(X,Y)$, and we care about how they behave *together*.

### Joint PMF (discrete)

:::def Joint PMF
$$p_{X,Y}(x,y)=P(X=x,\ Y=y)\qquad\qquad\sum_x\sum_y p_{X,Y}(x,y)=1$$
:::

It can be imagined as a **table**: rows are values of $X$, columns are values of $Y$, and every cell is a joint probability.

:::fig joint-grid
:::

:::ex Example: 80 students
:::fig students-table
:::

- **Joint:** $P(\text{passed \& studied})=\dfrac{36}{80}$
- **Marginal:** $P(\text{passed})=\dfrac{50}{80}$
- **Conditional:** $P(\text{failed}\mid\text{studied})=\dfrac{9}{45}$
:::

### Marginalization

Each cell gives only the joint distribution $p_{X,Y}(X=x,\ Y=y)\to$ it lives **inside** the cells.

But… suppose we want only $P(X=1)$: what is the probability that a randomly selected student studied? (Let $X=1$ mean studied, $Y=1$ mean passed.)

$X=1$ happens either when $Y=0$ or $Y=1$. So sum up the row/column with $X=1$:

$$P(X=1)=P(X=1,Y=1)+P(X=1,Y=0)=\frac{36}{80}+\frac{9}{80}=\frac{45}{80}$$

That's exactly the column total. Marginal $\to$ usually looks at **row totals or column totals** (they're written in the margins of the table, hence the name).

:::def Marginalization
$$P(X=x)=\sum_yP(X=x,\ Y=y)$$

To find the probability of $X=x$, add all joint probabilities containing $X=x$, **regardless of what $Y$ does**. Summing across $y$, we **sum out** $Y$.
:::

:::fig marginal-arrow
:::

:::viz students | Click a cell for the joint probability, a total for a marginal, or condition on a row or column: the cells in it are divided by its total and become a new distribution that sums to 1.
:::

### Continuous: joint PDF and marginals

For two continuous variables, the joint distribution is a **density** $f_{X,Y}(x,y)$. Marginalization works the same way, except we replace sums with integrals.

:::def Continuous marginals
To get the density of $X$ alone, integrate over every possible value of $y$:

$$f_X(x)=\int_{-\infty}^{\infty}f_{X,Y}(x,y)\,dy\qquad\qquad f_Y(y)=\int_{-\infty}^{\infty}f_{X,Y}(x,y)\,dx$$

Integrate out the unwanted variable.
:::

Same rule as before: a density is not a probability, probability comes from an area. With one variable that's $P(a\le X\le b)=\int_a^bf_X(x)\,dx$. Similarly, for two variables it's a **volume** over a region $R$:

$$\begin{aligned}P\big((X,Y)\in R\big)&=\iint_Rf_{X,Y}(x,y)\,dx\,dy\\\text{e.g.}\qquad P(0<X<1,\ 2<Y<4)&=\int_0^1\!\!\int_2^4f_{X,Y}(x,y)\,dy\,dx\end{aligned}$$

## 1.2.4 Conditional Probability Distributions

- **Marginal** $\to$ "Only care about $X$, ignore $Y$."
- **Conditional** $\to$ "Given something about $Y$, how does that change what I believe about $X$?"

:::def Conditional probability
$$P(A\mid B)=\frac{P(A\cap B)}{P(B)}\qquad\qquad\text{PMF notation:}\quad p_{X\mid Y}(x\mid y)=\frac{p_{X,Y}(x,y)}{p_Y(y)}$$

Joint divided by marginal.
:::

:::ex Example
$$P(X=1\mid Y=1)=\frac{P(X=1,\ Y=1)}{P(Y=1)}$$

We know $P(X=1,Y=1)=0.4$ and $P(Y=1)=0.6$. So

$$P(X=1\mid Y=1)=\frac{0.4}{0.6}=\frac23\qquad\text{and}\qquad P(X=0\mid Y=1)=\frac13$$

$\dfrac23+\dfrac13=1$. This **must** happen, because when we condition on $Y=1$, we now have a **new probability distribution** over the possible values of $X$.
:::

:::def Conditionals sum to 1
So for a fixed $Y=y$:

$$\sum_xp_{X\mid Y}(x\mid y)=1$$
:::

### Joint = Conditional $\times$ Marginal

Rearranging the definition:

$$P(X=x\mid Y=y)=\frac{P(X=x,\ Y=y)}{P(Y=y)}\quad\Longrightarrow\quad\underbrace{P(X=x\mid Y=y)}_{\text{conditional}}\cdot\underbrace{P(Y=y)}_{\text{marginal}}=\underbrace{P(X=x,\ Y=y)}_{\text{joint}}$$

:::def Product rule
$$P(X,Y)=P(X\mid Y)\,P(Y)=P(Y\mid X)\,P(X)$$
:::

Setting the two sides equal and dividing by $P(X)$ gives **Bayes' theorem**, which flips a conditional around:

$$\boxed{P(Y\mid X)=\frac{P(X\mid Y)\,P(Y)}{P(X)}}$$

### Law of total probability

The same split-into-subgroups idea as marginalization: if $B_1,\dots,B_k$ cut the sample space into pieces,

$$\boxed{P(A)=\sum_iP(A\mid B_i)\,P(B_i)}$$

Weight each subgroup's answer by how big the subgroup is. This is exactly what the denominator of Bayes' theorem is.

:::ex Bayes in action: a diagnostic test
Disease prevalence $P(D)=0.01$, sensitivity $P(+\mid D)=0.95$, false-positive rate $P(+\mid\bar D)=0.05$.

$$P(+)=\underbrace{0.95(0.01)}_{\text{sick}}+\underbrace{0.05(0.99)}_{\text{healthy}}=0.059\qquad\Rightarrow\qquad P(D\mid+)=\frac{0.95(0.01)}{0.059}\approx0.161$$

A positive test moves the probability from 1% to only $\approx$16% — nowhere near the 95% the sensitivity suggests. When a condition is rare, most positives come from the much larger healthy group.
:::

:::viz bayes | Each square is one person in 10,000. Move the three sliders; the counts of sick and healthy positives give $P(+)$ and $P(D\mid+)$.
:::

### Continuous conditionals

Same idea with densities:

$$f_{X\mid Y}(x\mid y)=\frac{f_{X,Y}(x,y)}{f_Y(y)}\qquad\text{so}\qquad f_{X,Y}(x,y)=f_{X\mid Y}(x\mid y)\cdot f_Y(y)$$

:::viz joint-density | Drag to orbit. Move the slice $y$: the cut through the joint density, rescaled by $f_Y(y)$, is $f_{X\mid Y}(x\mid y)$. Set $\rho=0$ and every slice has the same shape.
:::

:::ml Conditionals in ML
A classifier is really learning a conditional distribution: $P(\text{label}\mid\text{features})$. "Given these features, what do I believe about the label?"
:::

## 1.2.5 Independence

:::def Independence
$$P(X=x\mid Y=y)=P(X=x)$$

This means that, given the value of $y$, that information does **not** change the probability distribution of $x$. Hence they are independent.
:::

**Formula for independence (derivation):**

$$\begin{aligned}P(X=x\mid Y=y)&=\frac{P(X=x,\ Y=y)}{P(Y=y)}\\\text{if }X,Y\text{ independent:}\quad P(X=x\mid Y=y)&=P(X=x)\\\text{so}\quad P(X=x)&=\frac{P(X=x,\ Y=y)}{P(Y=y)}\end{aligned}$$

:::def Test for independence
$$P(X=x,\ Y=y)=P(X=x)\cdot P(Y=y)$$

$$\textbf{or}\quad P(X\mid Y)=P(X)\qquad\qquad\textbf{or}\quad P(X,Y)=P(X)P(Y)$$

Continuous: $f_{X,Y}(x,y)=f_X(x)\cdot f_Y(y)$.
:::

:::ex Example
:::cols figleft
:::col
:::fig indep-table
:::
:::
:::col
Last row/column are the marginals.\
**If $X=0$, $Y=0$:**\
$P(X=0,Y=0)=0.15$\
Check: $P(X)P(Y)=0.3\times0.5=0.15$ ✓\
If you check every other cell, it matches (e.g. $0.7\times0.5=0.35$ ✓). Hence $X,Y$ are **independent**.
:::
:::
:::

One cell failing is enough to break independence. All cells have to match.

:::ml Independence in ML
Naive Bayes assumes the features are independent given the class, so the joint splits into a product: $P(x_1,\dots,x_n\mid y)=\prod_iP(x_i\mid y)$. That's what makes it so cheap to compute.
:::

### Conditional independence

$X\perp Y\mid Z$ means: once you know $Z$, learning $Y$ tells you nothing more about $X$:

$$P(X,Y\mid Z)=P(X\mid Z)\,P(Y\mid Z).$$

E.g. two fair coin flips are independent outright: every cell has $p(x,y)=\tfrac14=\tfrac12\cdot\tfrac12$. Conditional independence is the version Naive Bayes above actually assumes: features independent *given the class*, not independent in general.

## 1.2.6 Expectation with Two Random Variables

Suppose every outcome gives us two values.

### Expectation of $X$ and $Y$ individually

$$E[X]=\sum_xx\,p_X(x)\qquad\qquad E[Y]=\sum_yy\,p_Y(y)$$

Since $p_X(x)=\sum_yp_{X,Y}(x,y)$ (marginalization), hence

$$E[X]=\sum_xx\cdot\sum_yp_{X,Y}(x,y)\qquad\text{same as}\qquad\boxed{E[X]=\sum_x\sum_yx\ p_{X,Y}(x,y)}$$

Note: we go through every $(x,y)$ pair, but only multiply the probability by $x$, because $X$ is what we are averaging. Similarly:

$$E[Y]=\sum_y\sum_xy\ p_{X,Y}(x,y)$$

### Expectation of a function of $X,Y$

Let $Z=g(X,Y)$, e.g. $Z=X+Y$ or $Z=XY$. So the expected value is:

:::def Expectation of $g(X,Y)$
$$E\big[g(X,Y)\big]=\sum_x\sum_yg(x,y)\ p_{X,Y}(x,y)$$

So, as always: **value** $\cdot$ **probability**, summed over every cell.
:::

:::ex Example: $E[XY]$
:::cols figleft
:::col
:::fig exy-table
:::
:::
:::col
We want $E[XY]$, so for each cell find $xy\cdot P(X=x,Y=y)$:

$$\begin{aligned}E[XY]&=(1)(1)(0.2)+(1)(2)(0.3)\\&\quad+(2)(1)(0.1)+(2)(2)(0.4)\\&=0.2+0.6+0.2+1.6\\E[XY]&=2.6\end{aligned}$$
:::
:::

$E[XY]$ is an **important quantity in covariance** (§1.2.7).
:::

### Expectation of a sum

$$\begin{aligned}E[X+Y]&=\sum_x\sum_y(x+y)\ p(x,y)\\&=\underbrace{\sum_x\sum_yx\cdot p(x,y)}_{E[X]}+\underbrace{\sum_x\sum_yy\cdot p(x,y)}_{E[Y]}\end{aligned}$$

:::def Linearity of expectation
$$E[X+Y]=E[X]+E[Y]$$

This is **regardless of independence**. It always works.
:::

### Expectation of a multiplication

:::def Product of independent variables
$$X\perp Y\implies E[XY]=E[X]\,E[Y]$$

**Only** when $X,Y$ are independent.
:::

**Why?** Independence means $p_{X,Y}(x,y)=p_X(x)\,p_Y(y)$, so

$$E[XY]=\sum_x\sum_yx\cdot y\ p_X(x)\,p_Y(y)$$

which separates into

$$\Big(\sum_xx\,p_X(x)\Big)\cdot\Big(\sum_yy\,p_Y(y)\Big)\qquad\text{thus}\qquad E[XY]=E[X]\,E[Y]$$

When they are *not* independent, $E[XY]$ and $E[X]E[Y]$ differ, and that gap is exactly what leads to **covariance**.

### Continuous version

With a joint PDF, sums become integrals:

$$E\big[g(X,Y)\big]=\int_{-\infty}^{\infty}\!\!\int_{-\infty}^{\infty}g(x,y)\ f_{X,Y}(x,y)\,dx\,dy\qquad\qquad E[XY]=\iint xy\ f_{X,Y}(x,y)\,dx\,dy$$

:::model Mental model
$$(X,Y)\;\longrightarrow\;\text{joint distribution }p_{X,Y}(x,y)\;\longrightarrow\;E\big[g(X,Y)\big]=\sum_x\sum_yg(x,y)\,p(x,y)$$

$$\longrightarrow\qquad E[X+Y]=E[X]+E[Y]\qquad\text{and}\qquad E[XY]$$
:::

## 1.2.7 Covariance and Correlation

### Covariance

Covariance measures how two variables move **together**: is $X$ above its mean when $Y$ is above its mean?

:::def Covariance
$$\operatorname{Cov}(X,Y)=E\big[(X-\mu_X)(Y-\mu_Y)\big]=E[XY]-E[X]\,E[Y]$$
:::

- This is the gap from the last section. If $X\perp Y$, then $E[XY]=E[X]E[Y]$, so $\operatorname{Cov}(X,Y)=0$.
- $\operatorname{Cov}(X,X)=E[X^2]-(E[X])^2=\operatorname{Var}(X)$. Variance is just covariance of a variable with itself.
- Careful: $\operatorname{Cov}=0$ does *not* mean independent. It only means no *linear* relationship.

:::ex Continuing the $E[XY]=2.6$ example
Marginals from the table: $P(X=1)=0.5,\ P(X=2)=0.5$ and $P(Y=1)=0.3,\ P(Y=2)=0.7$.

$$E[X]=1(0.5)+2(0.5)=1.5\qquad\qquad E[Y]=1(0.3)+2(0.7)=1.7$$

$$\operatorname{Cov}(X,Y)=E[XY]-E[X]E[Y]=2.6-(1.5)(1.7)=2.6-2.55=0.05$$

Positive, so they tend to increase together (slightly).
:::

### Correlation

The problem with covariance is that its magnitude depends on the **units** of $X$ and $Y$ (measure height in cm instead of m and the covariance jumps by $100\times$). So we divide by their std. dev.:

:::def Correlation
$$\rho_{X,Y}=\frac{\operatorname{Cov}(X,Y)}{\sigma_X\,\sigma_Y}\qquad\qquad\sigma_X=\sqrt{\operatorname{Var}(X)},\quad\sigma_Y=\sqrt{\operatorname{Var}(Y)}$$

Forcing correlation into

$$-1\le\rho_{X,Y}\le1$$
:::

:::fig rho-panels
:::

- $\rho>0\to$ increase together. $\rho=1\to$ perfect linear correlation.
- $\rho<0\to$ one goes up, the other goes down. $\rho=-1\to$ perfect negative linear.
- $\rho=0\to$ no linear relationship.

:::ex Same example
$$\operatorname{Var}(X)=E[X^2]-(E[X])^2=\big(1(0.5)+4(0.5)\big)-1.5^2=2.5-2.25=0.25\;\Rightarrow\;\sigma_X=0.5$$

$$\operatorname{Var}(Y)=\big(1(0.3)+4(0.7)\big)-1.7^2=3.1-2.89=0.21\;\Rightarrow\;\sigma_Y\approx0.458$$

$$\rho_{X,Y}=\frac{0.05}{(0.5)(0.458)}\approx0.22$$

A weak positive correlation. Now it's a unit-free number we can actually compare.
:::

:::viz table-lab | Drag the probability of any cell (the others rescale so the table still sums to 1). Everything below is recomputed from the table; the second preset is the independence example from §1.2.5.
:::

:::viz cov-scatter | Drag the points. Each rectangle is $(x-\mu_X)(y-\mu_Y)$: blue adds to the covariance, coral subtracts.
:::

:::ml Covariance & correlation in ML
- Two features with $|\rho|$ close to 1 carry almost the same info, which is the redundancy that lowers rank (§1.1.2).
- PCA works on the **covariance matrix** of the data: its eigenvectors are the directions of highest variation (§1.1.5).
:::

## 1.2.8 Conditional Expectation

- **Normal expectation:** $E[X]$ is the avg value across the **entire population**.
- **Conditional expectation:** what is $E[X]$ given $Y=y$?

:::def Conditional expectation
$$E[X\mid Y=y]=\sum_xx\ P(X=x\mid Y=y)$$

"The avg of $X$ inside the subgroup where $Y=y$."
:::

Compared with

$$E[X]=\sum_xx\ P(X=x)$$

the only difference is we replace the distribution with the **conditional distribution**.

**For continuous:**

$$E[X\mid Y=y]=\int_{-\infty}^{\infty}x\ f_{X\mid Y}(x\mid y)\,dx\qquad\text{where}\qquad f_{X\mid Y}(x\mid y)=\frac{f_{X,Y}(x,y)}{f_Y(y)}$$

### Law of total expectation

:::def Law of total expectation
**Overall average = weighted average of subgroup averages**

$$E[X]=\sum_yE[X\mid Y=y]\ P(Y=y)=E\big[E[X\mid Y]\big]$$
:::

:::ex Example
$X$ = exam score, $Y$ = studied or not.

- $E[X\mid Y=1]=80\to$ avg score of those who studied
- $E[X\mid Y=0]=60\to$ avg score of those who didn't study

Suppose $P(Y=1)=0.75$ and $P(Y=0)=0.25$. Then

$$E[X]=0.75(80)+0.25(60)=60+15=75$$

Each subgroup average gets weighted by how big that subgroup is.
:::

:::ml Conditional expectation in ML
A regression model predicting $y$ from $x$ is trying to learn $E[Y\mid X=x]$: the average target value for inputs like this one. Under MSE loss, that's exactly the best possible prediction.
:::

## 1.2.9 Putting It All Together: Discrete vs Continuous

Every idea in this part has a discrete version (sums) and a continuous version (integrals). Same logic, different tool.

|{stack}
| | Discrete | Continuous |
|---|---|---|
| **Single** | PMF: $p_X(x)=P(X=x)$ | PDF: $P(a\le X\le b)=\int_a^bf_X(x)\,dx$ |
| **CDF** | $F_X(x)=\sum_{t\le x}p_X(t)$ | $F_X(x)=\int_{-\infty}^xf_X(t)\,dt$ |
| **Mean** | $E[X]=\sum_xx\,p_X(x)$ | $E[X]=\int x\,f_X(x)\,dx$ |
| **Joint** | $P(X,Y)$ | density $f_{X,Y}(x,y)$ |
| **Marginals** | $P(X)=\sum_yP(X,y)$<br>$P(Y)=\sum_xP(x,Y)$ | $f_X(x)=\int f_{X,Y}(x,y)\,dy$<br>$f_Y(y)=\int f_{X,Y}(x,y)\,dx$ |
| **Conditionals** | $P(X\mid Y)=\dfrac{P(X,Y)}{P(Y)}$ | $f_{X\mid Y}(x\mid y)=\dfrac{f_{X,Y}(x,y)}{f_Y(y)}$ |
| **Independence** | does conditioning on $Y$ change $X$?<br>if not: $P(X\mid Y)=P(X)$,<br>$P(X,Y)=P(X)P(Y)$ | $f_{X,Y}(x,y)=f_X(x)\cdot f_Y(y)$ |
| $E[g(X,Y)]$ | $\sum_x\sum_yg(x,y)\,p(x,y)$ | $\iint g(x,y)\,f_{X,Y}(x,y)\,dx\,dy$ |

## 1.2.10 Quick Reference

|{stack}
| **Concept** | Formula / definition | Meaning |
|---|---|---|
| **PMF** | $p_X(x)=P(X=x),\ \sum p_X(x)=1$ | probability at one value |
| **PDF** | $P(a\le X\le b)=\int_a^bf_X\,dx$ | probability = area; $P(X=x)=0$ |
| **CDF** | $F_X(x)=P(X\le x)$ | accumulated probability up to $x$ |
| **PDF $\leftrightarrow$ CDF** | $f_X(x)=\frac{d}{dx}F_X(x)$ | integrate $\rightleftarrows$ differentiate |
| **Mean** | $E[X]=\sum x\,p_X(x)$ | value $\times$ how likely |
| **Variance** | $E[(X-\mu)^2]=E[X^2]-(E[X])^2$ | avg squared distance from mean |
| **Std. deviation** | $\sigma=\sqrt{\operatorname{Var}(X)}$ | spread in original units |
| **Joint** | $p_{X,Y}(x,y)=P(X=x,Y=y)$ | lives inside the cells |
| **Marginal** | $P(X=x)=\sum_yP(X=x,Y=y)$ | sum out $Y$; row/column totals |
| **Conditional** | $P(X\mid Y)=P(X,Y)/P(Y)$ | new distribution, sums to 1 |
| **Product rule** | $P(X,Y)=P(X\mid Y)P(Y)$ | joint = conditional $\times$ marginal |
| **Bayes** | $P(Y\mid X)=P(X\mid Y)P(Y)/P(X)$ | flip a conditional |
| **Independence** | $P(X,Y)=P(X)P(Y)$ | knowing $Y$ doesn't change $X$ |
| $E[g(X,Y)]$ | $\sum\sum g(x,y)\,p(x,y)$ | value $\cdot$ probability, every cell |
| **Linearity** | $E[X+Y]=E[X]+E[Y]$ | always, regardless of independence |
| **Product** | $E[XY]=E[X]E[Y]$ | only if $X\perp Y$ |
| **Covariance** | $E[XY]-E[X]E[Y]$ | move together? depends on units |
| **Correlation** | $\rho=\operatorname{Cov}/(\sigma_X\sigma_Y)$ | unit-free, $-1\le\rho\le1$ |
| **Cond. expectation** | $E[X\mid Y=y]=\sum x\,P(x\mid y)$ | avg inside the subgroup $Y=y$ |
| **Total expectation** | $E[X]=\sum_yE[X\mid Y=y]P(Y=y)$ | weighted avg of subgroup avgs |
| **Named dists** | $E=p,\ np,\ \lambda$ | yes/no; # successes; # events per window |
| **Normal rule** | 68–95–99.7 | within $\mu\pm\sigma,\ 2\sigma,\ 3\sigma$ |
| **Total probability** | $P(A)=\sum_iP(A\mid B_i)P(B_i)$ | weight subgroup answers by subgroup size |
| **Sample mean/var** | $\bar X=\frac1n\sum X_i$; $\ S^2=\frac{\sum(X_i-\bar X)^2}{n-1}$ | $\operatorname{Var}(\bar X)=\sigma^2/n$; $n{-}1$ makes $S^2$ unbiased |
