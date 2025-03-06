use crate::{Feature, MatrixDRows};
use faer::{linalg::solvers::Solve, Mat};
use faer_ext::{IntoFaer, IntoNalgebra};
use nalgebra::{DVector, SMatrix, SVector};
use std::sync::Arc;

#[cfg_attr(doc, katexit::katexit)]
/// Linear model containing its set of features
///
/// Defines the linear model ($y:\mathbb R^m \to\mathbb R$), where $y = \phi^T \beta$ with its
/// feature map $\phi:\mathbb R^m \to\mathbb R^n$ and coefficient $\beta \in \mathbb R^n$.
///
/// ```
/// use odesign::{Feature, FeatureFunction, LinearModel};
/// use nalgebra::{SVector};
/// use num_dual::DualNum;
/// use std::sync::Arc;
///
/// // Generic monomial (R^2 -> R) with derivatives by
/// // providing its feature function x^i * y^j only
/// #[derive(Feature)]
/// #[dimension = 2]
/// struct Monomial {
///     i: i32,
///     j: i32,
/// }
///
/// impl FeatureFunction<2> for Monomial {
///     fn f<D: DualNum<f64>>(&self, x: &SVector<D, 2>) -> D {
///         x[0].powi(self.i) * x[1].powi(self.j)
///     }
/// }
///
/// let monomial_a = Monomial { i: 0, j: 0 };
/// let monomial_b = Monomial { i: 1, j: 1 };
/// let monomial_c = Monomial { i: 2, j: 2 };
///
/// let lm = LinearModel::new(vec![
///     Arc::new(monomial_a),
///     Arc::new(monomial_b),
///     Arc::new(monomial_c),
/// ]);
/// ```
pub struct LinearModel<const D: usize> {
    /// Ordered list of features, building the feature map $\phi:\mathbb R^m \to\mathbb R^n$, where
    /// n is the number of features
    pub features: Vec<Arc<dyn Feature<D> + Send + Sync>>,
}

impl<const D: usize> LinearModel<D> {
    /// Creates linear model by providing the feature map $\phi:\mathbb R^m \to\mathbb R^n$.
    ///
    /// Since we are here in the context of optimal designs, we are not interested in providing the
    /// coefficient $\beta$ in order to define a linear model. We are moving that part to the
    /// methods.
    pub fn new(features: Vec<Arc<dyn Feature<D> + Send + Sync>>) -> Self {
        Self { features }
    }

    /// Returns the value $y = \phi^T \beta$.
    pub fn val(&self, x: &SVector<f64, D>, coefficients: &DVector<f64>) -> f64 {
        self.features
            .iter()
            .enumerate()
            .map(|(idx, f)| f.val(x) * coefficients[idx])
            .sum()
    }

    /// Returns the value and gradient $\nabla y$, where $(\nabla y)_i(x) =
    /// \frac{\partial y}{\partial x_i}(x), \forall i \in \[ m \]$.
    ///
    /// Since we are here in the context of nonlinear programming we are returning always partial
    /// results, such that we can work with the value and gradient. This makes especially sense
    /// in case where we use the Feature derive functionality, we are always computing the value
    /// in order to calculate the gradient. By using that method design, we can reduce duplicated computations.
    #[inline(always)]
    pub fn val_grad(
        &self,
        x: &SVector<f64, D>,
        coefficients: &DVector<f64>,
    ) -> (f64, SVector<f64, D>) {
        self.features
            .iter()
            .enumerate()
            .map(|(idx, t)| {
                let val_grad = t.val_grad(x);
                (
                    val_grad.0 * coefficients[idx],
                    val_grad.1 * coefficients[idx],
                )
            })
            .reduce(|(mut sum_val, mut sum_grad), (val, grad)| {
                sum_val += val;
                sum_grad += grad;
                (sum_val, sum_grad)
            })
            .unwrap()
    }
    /// Returns the value, gradient and hessian $H_y$, where $H_{ij} = \frac{\partial^2
    /// y}{\partial x_i \partial x_j}(x), \forall i,j \in \[ m \]$.
    ///
    /// Since we are here in the context of nonlinear programming we are returning always partial
    /// results, such that we can work with the value and gradient. This makes especially sense
    /// in case where we use the Feature derive functionality, we are always computing the value
    /// in order to calculate the gradient. By using that method design, we can reduce duplicated computations.
    #[inline(always)]
    pub fn val_grad_hes(
        &self,
        x: &SVector<f64, D>,
        coefficients: &DVector<f64>,
    ) -> (f64, SVector<f64, D>, SMatrix<f64, D, D>) {
        self.features
            .iter()
            .enumerate()
            .map(|(idx, t)| {
                let (val, grad, hes) = t.val_grad_hes(x);
                let c = coefficients[idx];
                (val * c, grad * c, hes * c)
            })
            .reduce(
                |(mut sum_val, mut sum_grad, mut sum_hes), (val, grad, hes)| {
                    sum_val += val;
                    sum_grad += grad;
                    sum_hes += hes;
                    (sum_val, sum_grad, sum_hes)
                },
            )
            .unwrap()
    }

    /// Returns the feature map $\phi(x)$.
    pub fn feature_vec(&self, x: &SVector<f64, D>) -> Mat<f64> {
        let mut feature_vec = Mat::<f64>::zeros(self.features.len(), 1);
        feature_vec
            .col_mut(0)
            .iter_mut()
            .enumerate()
            .for_each(|(idx, r)| {
                *r = self.features[idx].val(x);
            });
        feature_vec
    }

    /// Returns the vector of feature hessians where the k-th element contains $H_{\phi_k}$, where
    /// $(H_{\phi_k})_{ij} =\frac{\partial^2 \phi_k}{\partial x_i \partial x_j}(x), \forall i,j \in
    /// \[ m \]$.
    pub fn feature_vec_hessian(&self, x: &SVector<f64, D>) -> Vec<Mat<f64>> {
        self.features
            .iter()
            .map(|f| {
                f.val_grad_hes(x)
                    .2
                    .view_range(.., ..)
                    .into_faer()
                    .to_owned()
            })
            .collect::<Vec<Mat<f64>>>()
    }

    /// Returns the transposed design matrix $X^T$ with a given dataset $D \in \mathbb R^{m \times
    /// |D|}$ (the shape is chosen due to column orientated storage layout), where $(X_{\phi})_{ij}
    /// = \phi_j(x^{(i)}), i \in \[ |D| \], j \in \[ n\]$, where $x^{(i)}$ represents the i-th
    /// column of the given data matrix.
    pub fn design_t(&self, data: &MatrixDRows<D>) -> Mat<f64> {
        let no_features = self.features.len();
        let mut design_t = Mat::<f64>::zeros(no_features, data.ncols());
        design_t
            .col_iter_mut()
            .enumerate()
            .for_each(|(j, mut col)| {
                let x = data.column(j).into();
                for i in 0..col.nrows() {
                    col[i] = self.features[i].val(&x);
                }
            });
        design_t
    }

    /// Returns the normed global fisher information matrix $\mathcal{M} = \sum_{i=1}^{|D|} w_i \cdot
    /// \phi(x^{(i)}) \phi(x^{(i)})^T$ with a given dataset $D \in \mathbb R^{m \times
    /// |D|}$ and weights $w \in \mathbb R^{|D|}$.
    pub fn fim(&self, data: &MatrixDRows<D>, weights: &Mat<f64>) -> Mat<f64> {
        let design_t = self.design_t(data);
        let no_features = self.features.len();
        let mut fim = Mat::<f64>::zeros(no_features, no_features);
        for (c_idx, w) in weights.col(0).iter().enumerate() {
            fim += *w * design_t.col(c_idx) * design_t.col(c_idx).transpose();
        }
        fim
    }

    /// Returns the normed global fisher information matrix $\mathcal{M} = \sum_{i=1}^{|D|} w_i \cdot
    /// \phi(x^{(i)}) \phi(x^{(i)})^T$ with a given dataset $D \in \mathbb R^{m \times
    /// |D|}$ and weights $w \in \mathbb R^{|D|}$.
    pub fn fim_from_design_t(&self, design_t: &Mat<f64>, weights: &Mat<f64>) -> Mat<f64> {
        let no_features = self.features.len();
        let mut fim = Mat::<f64>::zeros(no_features, no_features);
        for (c_idx, w) in weights.col(0).iter().enumerate() {
            fim += *w * design_t.col(c_idx) * design_t.col(c_idx).transpose();
        }
        fim
    }

    /// Returns the transposed jacobian matrix $J^T \in \mathbb R^{m \times n}$, where $J_{ij}(x) = (\nabla \phi_j(x))_i$.
    pub fn jac_t(&self, x: &SVector<f64, D>) -> Mat<f64> {
        let mut m = Mat::<f64>::zeros(D, self.features.len());
        m.col_iter_mut().enumerate().for_each(|(idx, mut c)| {
            c.copy_from(
                self.features[idx]
                    .val_grad(x)
                    .1
                    .column(0)
                    .into_faer()
                    .col(0),
            );
        });
        m
    }

    /// Returns coefficient $\beta$ and root mean square error of the linear regression problem $Y = X
    /// \beta$ with design matrix $X$.
    pub fn fit(&self, data: &MatrixDRows<D>, y: &DVector<f64>) -> (DVector<f64>, f64) {
        let fy = y.view_range(.., ..).into_faer();
        let dm_t = self.design_t(data);
        let a = &dm_t * dm_t.transpose();
        let b = &dm_t * fy;
        let coeff = a.partial_piv_lu().solve(b);
        let rmse: f64 =
            (1. / data.shape().1 as f64).sqrt() * (dm_t.transpose() * &coeff - fy).norm_l2();
        (coeff.as_ref().into_nalgebra().column(0).into(), rmse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FeatureFunction, Result};
    use faer::mat;
    use nalgebra::{Matrix2, Vector2};
    use num_dual::DualNum;

    const EQ_EPS: f64 = 1e-8;
    const EQ_MAX_REL: f64 = 1e-8;

    #[derive(Feature)]
    #[dimension = 2]
    struct Monomial {
        i: i32,
        j: i32,
    }

    impl FeatureFunction<2> for Monomial {
        fn f<D: DualNum<f64>>(&self, x: &SVector<D, 2>) -> D {
            x[0].powi(self.i) * x[1].powi(self.j)
        }
    }

    fn get_polynomial() -> LinearModel<2> {
        let monomial_a = Monomial { i: 0, j: 0 };
        let monomial_b = Monomial { i: 1, j: 1 };
        let monomial_c = Monomial { i: 2, j: 2 };
        LinearModel::new(vec![
            Arc::new(monomial_a),
            Arc::new(monomial_b),
            Arc::new(monomial_c),
        ])
    }

    #[test]
    fn linear_model_value_gradient_hessian() -> Result<()> {
        let coeff = DVector::from_vec(vec![1., 2., 1.]);
        let x = Vector2::new(2., 1.);
        let polynomial = get_polynomial();
        // value, gradient, hessian
        let val_grad_hes_rslt = (9., Vector2::new(6., 12.), Matrix2::new(2., 10., 10., 8.));
        assert_eq!(polynomial.val(&x, &coeff), val_grad_hes_rslt.0);
        assert_eq!(
            polynomial.val_grad(&x, &coeff),
            (val_grad_hes_rslt.0, val_grad_hes_rslt.1)
        );
        assert_eq!(polynomial.val_grad_hes(&x, &coeff), val_grad_hes_rslt);
        Ok(())
    }

    #[test]
    fn linear_model_feature_vec() -> Result<()> {
        let polynomial = get_polynomial();
        let x = Vector2::new(2., 1.);
        // feature vec
        let feature_vec = polynomial.feature_vec(&x);
        assert_eq!(feature_vec, mat![[1.], [2.], [4.]]);
        // feature vec hessian
        let feature_vec_hessian = polynomial.feature_vec_hessian(&x);
        assert_eq!(
            feature_vec_hessian,
            vec![
                mat![[0., 0.], [0., 0.]],
                mat![[0., 1.], [1., 0.]],
                mat![[2., 8.], [8., 8.]]
            ]
        );
        Ok(())
    }

    #[test]
    fn design_matrix() -> Result<()> {
        let polynomial = get_polynomial();
        let data = MatrixDRows::from_vec(vec![1., 1., 2., 2.]);
        let design_t = polynomial.design_t(&data);
        assert_eq!(design_t, mat![[1., 1.], [1., 4.], [1., 16.]]);
        Ok(())
    }

    #[test]
    fn fisher_information_matrix() -> Result<()> {
        let polynomial = get_polynomial();
        let data = MatrixDRows::from_vec(vec![1., 1., 2., 2.]);
        let weights = mat![[2.], [2.]];
        let fim = polynomial.fim(&data, &weights);
        assert_eq!(
            fim,
            mat![[4., 10., 34.], [10., 34., 130.], [34., 130., 514.]]
        );
        Ok(())
    }

    #[test]
    fn fisher_information_matrix_from_design() -> Result<()> {
        let polynomial = get_polynomial();
        let data = MatrixDRows::from_vec(vec![1., 1., 2., 2.]);
        let design_t = polynomial.design_t(&data);
        let weights = mat![[2.], [2.]];
        let fim = polynomial.fim_from_design_t(&design_t, &weights);
        assert_eq!(
            fim,
            mat![[4., 10., 34.], [10., 34., 130.], [34., 130., 514.]]
        );
        Ok(())
    }

    #[test]
    fn jacobian() -> Result<()> {
        let polynomial = get_polynomial();
        let x = Vector2::new(2., 1.);
        let jac_t = polynomial.jac_t(&x);
        assert_eq!(jac_t, mat![[0., 1., 4.], [0., 2., 8.]]);
        Ok(())
    }

    #[test]
    fn linear_regression() -> Result<()> {
        let polynomial = get_polynomial();
        let data = MatrixDRows::from_vec(vec![0., 0., 1., 1., 2., 2.]);
        let y = DVector::from_vec(vec![2., 4., 22.]);
        let (coeff, rmse) = polynomial.fit(&data, &y);
        assert!(rmse < 1e-10);
        assert!(coeff.relative_eq(&DVector::from_vec(vec![2., 1., 1.]), EQ_EPS, EQ_MAX_REL));
        Ok(())
    }
}
