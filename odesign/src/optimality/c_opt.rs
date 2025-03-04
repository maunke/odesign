use crate::{Error, IntoSVector, LinearModel, MatrixDRows, NLPFunctionTarget, Optimality, Result};
use faer::{linalg::solvers::Solve, Mat};
use faer_ext::IntoFaer;
use nalgebra::{DVector, SVector};
use std::sync::Arc;

#[cfg_attr(doc, katexit::katexit)]
/// C-Optimality
///
/// C-Optimal Design of a linear model is defined as
///
/// $$ C_{opt} := c^T \mathcal{M}^{-1} c, \quad c_i \ge 0, \forall i \in 1, ..., n$$
///
/// with criterium $c$.
///
/// Since $\text{Var}[c^T \beta] \propto C_{opt}$ with linear model coefficient $\beta$
/// we can set $c^T = (0,0,...,0,1)$ in order to minimize the variance of the estimator
/// of $\beta_n$ in order to build an optimal design in sense of the null hypothesis
/// $H_0: \beta_n = 0$.
///
#[derive(Clone)]
pub struct COptimality<const D: usize> {
    linear_model: Arc<LinearModel<D>>,
    c: Arc<DVector<f64>>,
}

impl<const D: usize> COptimality<D> {
    /// Instantizes [COptimality]
    pub fn new(linear_model: Arc<LinearModel<D>>, c: Arc<DVector<f64>>) -> Result<Self> {
        if linear_model.features.len() != c.nrows() {
            return Err(Error::ShapeMismatch {
                mat1: "features",
                mat2: "c",
                dim1: 0,
                dim2: 0,
                shape1: (linear_model.features.len(), 1),
                shape2: c.shape(),
            });
        }
        Ok(Self { linear_model, c })
    }
}

impl<const D: usize> Optimality<D> for COptimality<D> {
    fn matrix_mean(&self, supp: Arc<MatrixDRows<D>>) -> Arc<dyn NLPFunctionTarget + Send + Sync> {
        Arc::new(CMatrixMeans::new(
            self.linear_model.clone(),
            supp,
            self.c.clone(),
        ))
    }
    fn dispersion_function(
        &self,
        supp: Arc<MatrixDRows<D>>,
        weights: Mat<f64>,
        _x_id: usize,
    ) -> Arc<dyn NLPFunctionTarget + Send + Sync> {
        Arc::new(CDispersionFunction::new(
            self.linear_model.clone(),
            weights,
            supp,
            self.c.clone(),
        ))
    }
    fn measure(&self, weights: &Mat<f64>, supp: Arc<MatrixDRows<D>>) -> f64 {
        self.matrix_mean(supp).val(weights).exp()
    }
}

pub struct CMatrixMeans<const D: usize> {
    linear_model: Arc<LinearModel<D>>,
    design_t: Mat<f64>,
    design: Mat<f64>,
    supp: Arc<MatrixDRows<D>>,
    c: Mat<f64>,
}

impl<const D: usize> CMatrixMeans<D> {
    pub fn new(
        linear_model: Arc<LinearModel<D>>,
        supp: Arc<MatrixDRows<D>>,
        c: Arc<DVector<f64>>,
    ) -> Self {
        let design_t = linear_model.design_t(&supp);
        let design = design_t.transpose().to_owned();
        let c = c.view_range(.., ..).into_faer().to_owned();
        Self {
            linear_model,
            design_t,
            design,
            supp,
            c,
        }
    }

    #[inline(always)]
    fn fim_e_n(&self, x: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
        let fim = self.linear_model.fim(&self.supp, x);
        let fim_e_n = fim.partial_piv_lu().solve(&self.c);
        (fim, fim_e_n)
    }
}

impl<const D: usize> NLPFunctionTarget for CMatrixMeans<D> {
    fn val(&self, x: &Mat<f64>) -> f64 {
        let (_, fim_e_n) = self.fim_e_n(x);
        let phi: f64 = (self.c.transpose() * fim_e_n)[(0, 0)];
        phi.ln()
    }
    fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
        let (_, fim_e_n) = self.fim_e_n(x);
        let theta = &self.design * &fim_e_n;
        let phi: f64 = (self.c.transpose() * fim_e_n)[(0, 0)];
        let val = phi.ln();
        let factor = -1. / phi;
        let mut grad = Mat::<f64>::zeros(x.nrows(), 1);
        for row in 0..grad.nrows() {
            let v = theta[(row, 0)];
            grad[(row, 0)] = factor * v * v;
        }
        (val, grad)
    }
    fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
        let (fim, fim_e_n) = self.fim_e_n(x);
        let theta = &self.design * &fim_e_n;
        let phi: f64 = (self.c.transpose() * fim_e_n)[(0, 0)];
        let val = phi.ln();
        let factor = -1. / phi;
        let mut grad = Mat::<f64>::zeros(x.nrows(), 1);
        for row in 0..grad.nrows() {
            let v = theta[(row, 0)];
            grad[(row, 0)] = factor * v * v;
        }
        let factor = 2. / phi;
        let theta_outer = factor * theta.col(0) * theta.col(0).transpose();
        let grad_outer = grad.col(0) * grad.col(0).transpose();
        let mut hes = &self.design * fim.partial_piv_lu().solve(&self.design_t);
        for col in 0..hes.ncols() {
            for row in col..hes.nrows() {
                let v = hes[(row, col)];
                let t_v = theta_outer[(row, col)];
                hes[(row, col)] = v * t_v;
            }
        }
        hes += grad_outer;
        (val, grad, hes)
    }
}

pub struct CDispersionFunction<const D: usize> {
    linear_model: Arc<LinearModel<D>>,
    fim_e_n: Mat<f64>,
    c: Mat<f64>,
}

impl<const D: usize> CDispersionFunction<D> {
    pub fn new(
        linear_model: Arc<LinearModel<D>>,
        weights: Mat<f64>,
        supp: Arc<MatrixDRows<D>>,
        c: Arc<DVector<f64>>,
    ) -> Self {
        let fim = linear_model.fim(&supp, &weights);
        let fim_lu = fim.partial_piv_lu();
        let c = c.view_range(.., ..).into_faer().to_owned();
        let fim_e_n = fim_lu.solve(&c);
        Self {
            linear_model,
            fim_e_n,
            c,
        }
    }

    #[inline(always)]
    pub fn phi_theta(&self, x: &SVector<f64, D>) -> (f64, f64) {
        let feature_vec = self.linear_model.feature_vec(x);
        let theta: f64 = (feature_vec.transpose() * &self.fim_e_n)[(0, 0)];
        let phi: f64 = (self.c.transpose() * &self.fim_e_n)[(0, 0)];
        (phi, theta)
    }

    #[inline(always)]
    pub fn jac_fim_en(&self, x: &SVector<f64, D>) -> Mat<f64> {
        let jac_t = self.linear_model.jac_t(x);
        jac_t * &self.fim_e_n
    }
}

impl<const D: usize> NLPFunctionTarget for CDispersionFunction<D> {
    fn val(&self, x: &Mat<f64>) -> f64 {
        let x_svector = x.into_svector();
        let (phi, theta) = self.phi_theta(&x_svector);
        phi - theta.powi(2)
    }
    fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
        let x_svector = x.into_svector();
        let (phi, theta) = self.phi_theta(&x_svector);
        let jac_fim_e_n = self.jac_fim_en(&x_svector);
        let val = phi - theta.powi(2);
        let grad = -theta * &jac_fim_e_n;
        (val, grad)
    }
    fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
        let x_svector = x.into_svector();
        let feature_vec = self.linear_model.feature_vec(&x_svector);
        let theta: f64 = (feature_vec.transpose() * &self.fim_e_n)[(0, 0)];
        let phi: f64 = (self.c.transpose() * &self.fim_e_n)[(0, 0)];
        let jac_fim_e_n = self.jac_fim_en(&x_svector);
        let val = phi - theta.powi(2);
        let grad = -theta * &jac_fim_e_n;
        let mut hes = -theta * &jac_fim_e_n * jac_fim_e_n.transpose();
        self.linear_model
            .feature_vec_hessian(&x_svector)
            .iter()
            .enumerate()
            .for_each(|(feat_id, feature_hes)| {
                let factor = self.fim_e_n[(feat_id, 0)];
                hes -= factor * feature_hes;
            });
        (val, grad, hes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_nlp_target_consistency, Feature, FeatureFunction, Result};
    use nalgebra::SVector;
    use num_dual::DualNum;

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

    fn get_linear_model() -> LinearModel<2> {
        let monom_a = Monomial { i: 1, j: 1 };
        let monom_b = Monomial { i: 1, j: 2 };
        LinearModel::new(vec![Arc::new(monom_a), Arc::new(monom_b)])
    }

    #[test]
    fn c_matrix_mean_consistency() -> Result<()> {
        let linear_model: Arc<_> = get_linear_model().into();
        let supp: Arc<_> = MatrixDRows::from_vec(vec![1., 2., 3., 4.]).into();
        let c = DVector::from_vec(vec![0., 1.]);
        let ccrit = CMatrixMeans::new(linear_model, supp, c.into());
        let x = Mat::<f64>::ones(2, 1);
        assert_nlp_target_consistency!(ccrit, &x);
        Ok(())
    }

    #[test]
    fn c_dispersion_function_consistency() -> Result<()> {
        let linear_model: Arc<_> = get_linear_model().into();
        let supp: Arc<_> = MatrixDRows::from_vec(vec![1., 2., 3., 4.]).into();
        let weights = Mat::<f64>::ones(2, 1);
        let c = DVector::from_vec(vec![0., 1.]);
        let ccrit = CDispersionFunction::new(linear_model, weights, supp, c.into());
        let x = Mat::<f64>::ones(2, 1);
        assert_nlp_target_consistency!(ccrit, &x);

        Ok(())
    }

    #[test]
    fn test_c_design_error() -> Result<()> {
        let linear_model: Arc<_> = get_linear_model().into();
        let c = DVector::from_vec(vec![0.]);
        let c_opt = COptimality::new(linear_model, c.into());
        let c_opt_err = Error::ShapeMismatch {
            mat1: "features",
            mat2: "c",
            dim1: 0,
            dim2: 0,
            shape1: (2, 1),
            shape2: (1, 1),
        };
        assert_eq!(c_opt.err(), Some(c_opt_err));
        Ok(())
    }
}
