# Optimalities

## Matrix Means

The best known class of information functions was introduced by Kiefer, 1974
(doi: [10.1214/aos/1176342810](https://doi.org/10.1214/aos/1176342810)). Each
information function is described by a hyperparameter δ ∈ (-∞, 1], which can be
categorized as follows into the class of matrix means ∆δ : PD (n) → R+ (where
PD(n) stands for a positive definite symmetric n x n matrix) as follows

$$
\Delta_{\delta}\left(\mathcal{M}(\xi)\right) = \left[\frac{1}{n}\text{tr}{\mathcal{M}(\xi)^\delta}\right]^{\frac{1}{\delta}} \quad \text{,}
$$

with the fisher information matrix M and the design ξ.

Right now the popular D-Optimalty with δ = 0 is part of `odesign`. Additionally
the C-Optimality with

$$
\Delta_c(\xi) := c^T \mathcal{M}^{-1}(\xi) c
$$

is implemented.

## Custom optimality

Beside the predefined optimalities, you can create your custom ones and/or
create a weighted sum of different optimalities, e.g. CD-Optimality.

Lets consider a simple example where we define a to be maximized
costs-efficiency-optimality that is concave and sums up the negative weighted
sum of the norm of each support vector

$$
\Delta\_{\text{costs}} := \exp\big(-\sum_{i \in [1, .., n]} w_i \cdot || x_i || \big)
$$

with n support vectors x and theirs weights w (see the
[custom-optimality example here](https://git.sr.ht/~maunke/odesign/tree/main/item/odesign-examples/examples/custom-optimality/main.rs)).

Since the solver minimizes the sum of negative log of the desired optimalities,
we formulate the custom matrix means derivatives as follows:

Value:

$$
-\log \Delta\_{\text{costs}} = \sum_{i \in [1, .., n]} w_i \cdot || x_i ||
$$

Gradient:

$$
\frac{\partial -\log \Delta\_{\text{costs}}}{\partial w_i} = || x_i ||
$$

Hessian:

$$
\frac{\partial^2 -\log \Delta\_{\text{costs}}}{\partial w_i \partial w_j} = 0
$$

The same principle applies to the dispersion function, where we will derive to
the support vector.
