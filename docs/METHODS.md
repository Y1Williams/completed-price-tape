# Methods for Figure 1

This note describes the deterministic calculation underlying the original Figure 1. The figure compares the largest Fisher information obtainable from a scalar execution cost and from a completed price tape, with a separate calibration design for each observation. The formulas below apply to the continuous class of bounded execution rates. Their displayed decimal values are numerical evaluations.

## Model and information criteria

Set the execution horizon, order size, and Brownian volatility to $T=Q=\sigma=1$, the relative rate cap to $R=2$, and the independent cost disturbance to $\tau=0$. The admissible controls are

$$
\mathcal V=\left\{v\in L^\infty(0,1):0\leq v\leq2,\quad\int_0^1v(t)\,dt=1\right\}.
$$

The recovery family is

$$
g_\theta(b)=\frac23\left[1+\theta\frac{5-4b}{3}\right],
\qquad b\in[1/2,2],\quad |\theta|\leq1/2.
$$

It has unit mass and a positive density. Define its parameter derivative and response kernel by

$$
h(b)=\frac29(5-4b),\qquad
F_h(b)=\int_{1/2}^b h(c)\,dc=\frac49(b-1/2)(2-b),
\qquad H(t)=\int_{1/2}^2h(b)e^{-bt}\,db.
$$

The zero integral of $h$ gives $H(0)=0$. Integration by parts gives $H(t)=t\int F_h(b)e^{-bt}\,db>0$ for $t>0$. Here and below an integral in the recovery rate runs over $[1/2,2]$.

The price tape is $Y_v(t)=m_{\theta,v}(t)+W(t)$ on $[0,1+U]$, where

$$
m_{\theta,v}(t)=\int_0^{\min(t,1)}\int_{1/2}^2g_\theta(b)e^{-b(t-s)}\,db\,v(s)\,ds.
$$

The cost is $C_v=\int_0^1v(t)Y_v(t)\,dt$. Its noise shares the Brownian path observed in the tape. Write

$$
\begin{aligned}
r_v(t)&=\partial_\theta m_{\theta,v}(t)
       =\int_0^{\min(t,1)}H(t-s)v(s)\,ds,\\
\dot r_v(t)&=\int_0^{\min(t,1)}H'(t-s)v(s)\,ds,\\
d(v)&=\frac12\int_0^1\int_0^1H(|s-t|)v(s)v(t)\,ds\,dt,\\
F_v(t)&=\int_t^1v(s)\,ds,\qquad
V_v=\int_0^1F_v(t)^2\,dt.
\end{aligned}
$$

Gaussian shift information then gives

$$
J_C(v)=\frac{d(v)^2}{V_v},\qquad
J_{CY,U}(v)=\int_0^{1+U}\dot r_v(t)^2\,dt.
$$

These criteria do not depend on $\theta$. Since $\tau=0$, the cost is a known functional of the tape, so observing the cost together with the tape has the tape's information. The plotted target is $J_C^\star/J_{CY,U}^\star$, where each star denotes a supremum over $\mathcal V$.

## Time blocks and the circles

For $n$ equal blocks $I_i=[i/n,(i+1)/n]$, with $i=0,\ldots,n-1$, let $w_i$ be the quantity executed on block $i$. The rate there is $nw_i$, and the constraints are $0\leq w_i\leq2/n$ and $\sum_iw_i=1$. Define

$$
\begin{aligned}
A_{ij}&=\frac{n^2}{2}\int_{I_i}\int_{I_j}H(|s-t|)\,ds\,dt,\\
V_{ij}&=n^2\int_{I_i}\int_{I_j}\min(s,t)\,ds\,dt,\\
M_i(t)&=n\left[H((t-i/n)_+)-H((t-(i+1)/n)_+)\right],\\
P_{ij}(U)&=\int_0^{1+U}M_i(t)M_j(t)\,dt.
\end{aligned}
$$

Consequently $d(v)=w^\mathsf TAw$, $V_v=w^\mathsf TVw$, and $J_{CY,U}(v)=w^\mathsf TP(U)w$. The matrix $V$ uses block averages of the Brownian covariance. With $\Delta=1/n$ and block centers $c_i=(i+1/2)/n$, its off diagonal entries are $\min(c_i,c_j)$ and its diagonal entries are $i/n+\Delta/3$.

The block average of $e^{-b|s-t|}$ in $A$ is evaluated analytically. It equals

$$
e^{-b|c_i-c_j|}\left[\frac{\sinh(b\Delta/2)}{b\Delta/2}\right]^2
\quad(i\ne j),\qquad
\frac{2(b\Delta+e^{-b\Delta}-1)}{(b\Delta)^2}
\quad(i=j).
$$

[model.py](../src/tape_example/model.py) uses 64 Gauss–Legendre nodes for the recovery rate integral. It evaluates the time integral defining $P$ with 12 nodes in each execution block and 80 nodes on $[1,1+U]$. Subtracting one inside the exponential in the implementation of $H$ uses $\int h=0$ to reduce cancellation near zero.

[numerics.py](../src/tape_example/numerics.py) maximizes $(w^\mathsf TAw)^2/(w^\mathsf TVw)$ and $w^\mathsf TP(U)w$ separately using SLSQP, analytic gradients, and fixed initial controls. It records calculations with $n=16,32,64$. Figure 1 shows selected ratios for $n=64$. Each circle is a ratio of two attained objective values. Multistart local optimization supplies no global optimality certificate. Since both objectives are optimized numerically, their ratio has no general one sided relation to the exact ratio of suprema.

## A continuous upper bound for cost information

A feasible candidate $v_0$ immediately gives the lower bound $c_-=J_C(v_0)$. A supporting functional gives a continuous upper bound independent of whether the candidate is optimal.

First, the rate cap implies $F_v(t)\geq(1-2t)_+$, hence $V_v\geq1/6$. Also, integration by parts in the recovery rate gives

$$
H''(t)=-\int bF_h(b)(2-bt)e^{-bt}\,db\leq0,\qquad
H'''(t)=\int b^2F_h(b)(3-bt)e^{-bt}\,db\geq0
$$

on $[0,1]$. Moreover, $H'(1/2)=\int F_h(b)(1-b/2)e^{-b/2}\,db>0$. If $f\in L^\infty(0,1)$, $\int_0^1f=0$, and $Z(t)=\int_0^tf(s)\,ds$, two integrations by parts give

$$
\begin{aligned}
\iint H(|s-t|)f(s)f(t)\,ds\,dt
&=-2H'(0)\int Z^2-\iint H''(|s-t|)Z(s)Z(t)\,ds\,dt\\
&\leq-\int_0^1[H'(t)+H'(1-t)]Z(t)^2\,dt\\
&\leq-2H'(1/2)\int_0^1Z(t)^2\,dt\leq0.
\end{aligned}
$$

The first inequality uses $2Z(s)Z(t)\leq Z(s)^2+Z(t)^2$ and $H''\leq0$. The second uses the convexity of $H'$. Thus $d$ is concave on controls of fixed total mass.

Set $d_0=d(v_0)$, $V_0=V_{v_0}$, $s_0=d_0/\sqrt{V_0}$, and

$$
k_0(t)=\int_0^1\min(t,r)v_0(r)\,dr,\qquad
a_0(t)=\int_0^1H(|t-r|)v_0(r)\,dr-\frac{d_0}{V_0}k_0(t).
$$

Since $\sqrt{V_u}$ is the norm of a linear map of $u$, the functional $d(u)-s_0\sqrt{V_u}$ is concave. It vanishes at $v_0$, has derivative $a_0$ there, and satisfies $\int a_0v_0=d_0$. Therefore, for any

$$
\varepsilon\geq\sup_{u\in\mathcal V}\int_0^1a_0(t)u(t)\,dt-d_0,
$$

the supporting inequality and $V_u\geq1/6$ imply

$$
c_-\leq J_C^\star\leq c_+:=(s_0+\sqrt6\,\varepsilon)^2.
$$

To evaluate the linear supremum, observe that $|H'|\leq1/4$ on $[0,1]$. Indeed, $H'$ decreases from $1/4$ to $(11e^{-1/2}-50e^{-2})/9>-1/4$. Since $k_0'=F_{v_0}\in[0,1]$, a Lipschitz constant for $a_0$ is $L=1/4+d_0/V_0$. On an even midpoint mesh of size $m$, let $S_m$ contain the largest $m/2$ values of $a_0$. The corresponding step function attains its linear maximum by assigning rate two to those cells. The uniform midpoint approximation error is at most $L/(2m)$, and every admissible control has mass one. Hence

$$
\varepsilon_m=\max\left\{0,\frac2m\sum_{j\in S_m}a_0((j-1/2)/m)+\frac{L}{2m}-d_0\right\}
$$

is a valid slack for the exact quantities.

The calculation refines the cost candidate from 64 to 128 intervals and evaluates the envelope in [bounds.py](../src/tape_example/bounds.py) with $m=65\,536$. The supporting potential uses 96 recovery rate nodes. A second evaluation uses $m=32\,768$ and 64 nodes. This comparison checks numerical consistency and the expected decrease in the mesh contribution.

## Tape bounds and the optimal pulse for $U\geq2$

Let $v_p=2\mathbf1_{[0,1/2]}$, $p_U=J_{CY,U}(v_p)$, and $p_\infty=J_\infty(v_p)$, where $J_\infty(v)=\int_0^\infty\dot r_v(t)^2\,dt$. Feasibility gives $p_U\leq J_{CY,U}^\star$. To bound the latter from above, write

$$
k_\infty(\ell)=\int_0^\infty H'(x)H'(x+\ell)\,dx,\qquad
J_\infty(v)=\iint k_\infty(|s-t|)v(s)v(t)\,ds\,dt.
$$

This autocorrelation kernel decreases on $[0,1]$. A direct sign estimate verifies the claim. Define

$$
N(c)=-\int\frac{bh(b)}{b+c}\,db
     =c\int\frac{F_h(b)}{(b+c)^2}\,db>0.
$$

Then $k_\infty'(\ell)=\int c^2h(c)N(c)e^{-c\ell}\,dc$. The integrand at zero has positive mass $P$ below $5/4$ and negative mass above it. Since $c/(b+c)^2\leq1/(4b)$,

$$
P\leq\overline P=\frac{19}{128}\left(\frac5{24}-\frac29\log2\right),
\qquad k_\infty'(0)=-\frac{H'(0)^2}{2}=-\frac1{32}.
$$

Here $\int_{1/2}^{5/4}c^2h(c)\,dc=19/128$ and $\int F_h(b)/(4b)\,db=5/24-2\log2/9$. The magnitude of the negative part is $P+1/32$. Bounding the exponential separately on the positive and negative parts yields

$$
k_\infty'(\ell)\leq e^{-2\ell}\left[(e^{3\ell/2}-1)\overline P-\frac1{32}\right]<0
\qquad(0\leq\ell\leq1).
$$

The bracket is increasing and remains negative at one. For example, $e^{3/2}<9/2$ and $\log2>69/100$ already prove this strict sign.

A quantile argument gives the required rearrangement bound directly under the rate cap. Let $q_v(u)$ be the time at which cumulative execution first reaches quantity $u\in(0,1)$. Then

$$
q_v(w)-q_v(u)\geq\frac{w-u}{2}\quad(u<w),\qquad
q_v(u)\geq u/2,\qquad
\int f(t)v(t)\,dt=\int_0^1f(q_v(u))\,du.
$$

Since all time separations lie in $[0,1]$, monotonicity of $k_\infty$ gives

$$
J_\infty(v)=\int_0^1\int_0^1k_\infty(|q_v(u)-q_v(w)|)\,du\,dw
\leq\int_0^1\int_0^1k_\infty(|u-w|/2)\,du\,dw=p_\infty.
$$

Thus $p_U\leq J_{CY,U}^\star\leq p_\infty$ for every $U\geq0$.

For $U\geq2$, the earliest pulse also minimizes the information omitted after observation ends. Direct integration gives

$$
H''(2)=\frac7{36}(22-e^3)e^{-4}>0,\qquad
H'(2)=-\frac32e^{-4}<0.
$$

The single sign change of $b^2h(b)$ at $5/4$ implies

$$
H''(t)\geq e^{-(5/4)(t-2)}H''(2)>0\qquad(t\geq2).
$$

Since $H'(t)\to0$, it is negative and increasing on this interval. For $t\geq1+U$ and $s\in[0,1]$, the argument $t-s$ is at least two, so $H'(t-s)$ is negative and decreases as $s$ increases. The quantile inequality therefore gives

$$
\dot r_v(t)=\int_0^1H'(t-q_v(u))\,du
\leq\int_0^1H'(t-u/2)\,du=\dot r_{v_p}(t)<0.
$$

Squaring and integrating shows that the pulse minimizes the omitted tail. Combining this with its infinite horizon maximum proves

$$
J_{CY,U}(v)=J_\infty(v)-\int_{1+U}^\infty\dot r_v(t)^2\,dt
\leq J_{CY,U}(v_p)=p_U,\qquad U\geq2.
$$

Hence $J_{CY,U}^\star=p_U$ for $U\geq2$. The proof makes no claim of pulse optimality for shorter windows.

The pulse values use recovery rate quadrature after integrating time analytically. If $\psi(b)=h(b)(1-e^{-b/2})$, then

$$
\begin{aligned}
p_\infty&=4\iint\frac{h(b)h(c)[1-e^{-(b+c)/2}]}{b+c}\,db\,dc
             +4\iint\frac{\psi(b)\psi(c)}{b+c}\,db\,dc,\\
p_U&=p_\infty-4\iint\frac{\psi(b)\psi(c)e^{-(b+c)(U+1/2)}}{b+c}\,db\,dc.
\end{aligned}
$$

The implementation uses 96 Gauss–Legendre nodes per recovery rate variable. At $U=2$, it compares this expression with $w_p^\mathsf TP(2)w_p$ from the block matrices.

## The blue curve and numerical interpretation

Combining the bounds gives

$$
\frac{c_-}{p_\infty}\leq\frac{J_C^\star}{J_{CY,U}^\star}
\leq\min\{1,c_+/p_U\}.
$$

The upper bound also uses $J_C(v)\leq J_{CY,U}(v)$, which follows because the cost is observed with the tape. For $U\geq2$, the lower endpoint improves to $c_-/p_U$. The blue curve evaluates the upper endpoint. Its calculation uses the continuous cost bound and the explicit pulse, independently of the circles.

At $U=2$, the optimized information ratio is approximately $0.316$. The figure displays $[0.315,0.317]$ by rounding the lower numerical endpoint down and the upper endpoint up to three decimal places. The [reference results](../data/reference/continuum_results.json) retain the full numerical precision of the cost bounds, pulse information, and ratio endpoints.

The analytic inequalities above concern exact integrals and feasible controls. The implementation uses floating point arithmetic, Gaussian quadrature, and optimization tolerances. Its `PAD = 2e-10` expands computed endpoints and perturbs ratio denominators. The code does not derive a total numerical error bound for this padding. The resulting decimals, including the outward display rounding, are numerical evaluations of analytic bound formulas and do not constitute rigorous floating point enclosures. The field `tape_exact_at_U2` stores the numerical evaluation of the analytically optimal pulse formula.

The saved candidate on 128 intervals has a mass residual of about $5\times10^{-15}$. All saved grid optimizer statuses report success. The current optimizer accepts a result only when the solver reports success, the objective is finite, the mass residual is at most $10^{-9}$, and violations of the weight bounds are at most $10^{-10}$. It records diagnostics for every initial control. Quadrature cross checks, finer meshes, and successful optimizer exits provide numerical consistency evidence.

The reproduced information calculation alone does not establish the regularity assumptions needed to translate an information ratio into a calibration episode requirement for this particular recovery family.
