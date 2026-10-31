# Things noticed in the notes PDF (nothing on the site was changed; Faris decides)

| Where | What the notes say | What the numbers give | Status |
|---|---|---|---|
| §1.1.4, orthogonal matrix example (PDF p. 12) | $Q=\frac{1}{\sqrt2}\begin{bmatrix}1&1\\-1&1\end{bmatrix}$ "rotates the plane by $45^\circ$" | it rotates **clockwise** by 45° ($Qe_1=(1,-1)/\sqrt2$, angle $-45^\circ$) | wording question, kept verbatim |
| §2.3.5, lecture examples table (PDF p. 73) | "High P, low R": TP 10, FN 40, FP 2 → F1 = **0.320** | $P=10/12=0.833$, $R=10/50=0.2$, $F_1=2PR/(P+R)=0.3226$ → **0.323** | kept verbatim; `scripts/check/2_3.py` prints it |
| §2.3.5, same table | "High R, low P": F1 = 0.666 | $0.6667$ (rounds to 0.667; 0.666 is truncated) | rounding only, kept verbatim |
