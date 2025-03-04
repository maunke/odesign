use crate::{IntoSVector, LinearModel, MatrixDRows, NLPFunctionTarget, Optimality};
use faer::{
    linalg::solvers::{PartialPivLu, Solve},
    Mat,
};
use nalgebra::SVector;
use std::sync::Arc;

#[cfg_attr(doc, katexit::katexit)]
/// D-Optimality is defined as determinant reciprocal to the power of n  of the
/// fisher-information matrix.
///
/// $$ D_{opt} := {(\det \mathcal{M})}^{\frac{1}{n}} $$
///
/// with fisher-information matrix $\mathcal{M}$ and number of coefficients $n$.
///
#[derive(Clone)]
pub struct DOptimality<const D: usize> {
    linear_model: Arc<LinearModel<D>>,
}

impl<const D: usize> DOptimality<D> {
    /// Instantizes [DOptimality]
    pub fn new(linear_model: Arc<LinearModel<D>>) -> Self {
        Self { linear_model }
    }
}

impl<const D: usize> Optimality<D> for DOptimality<D> {
    fn matrix_mean(&self, supp: Arc<MatrixDRows<D>>) -> Arc<dyn NLPFunctionTarget + Send + Sync> {
        Arc::new(DMatrixMean::new(self.linear_model.clone(), supp))
    }
    fn dispersion_function(
        &self,
        supp: Arc<MatrixDRows<D>>,
        weights: Mat<f64>,
        _x_id: usize,
    ) -> Arc<dyn NLPFunctionTarget + Send + Sync> {
        Arc::new(DDispersionFunction::new(
            self.linear_model.clone(),
            weights,
            supp,
        ))
    }
    fn measure(&self, weights: &Mat<f64>, supp: Arc<MatrixDRows<D>>) -> f64 {
        let m_mean = self.matrix_mean(supp);
        let val = m_mean.val(weights);
        (-val)
            .exp()
            .powf(1. / (self.linear_model.features.len() as f64))
    }
}

pub struct DMatrixMean<const D: usize> {
    linear_model: Arc<LinearModel<D>>,
    design_t: Mat<f64>,
    design: Mat<f64>,
    supp: Arc<MatrixDRows<D>>,
}

impl<const D: usize> DMatrixMean<D> {
    pub fn new(linear_model: Arc<LinearModel<D>>, supp: Arc<MatrixDRows<D>>) -> Self {
        let design_t = linear_model.design_t(&supp);
        let design = design_t.transpose().to_owned();
        Self {
            linear_model,
            design_t,
            design,
            supp,
        }
    }
    #[inline(always)]
    fn fim_det(&self, x: &Mat<f64>) -> (Mat<f64>, f64) {
        let fim = self.linear_model.fim(&self.supp, x);
        let det = fim.determinant();
        (fim, det)
    }
    #[inline(always)]
    fn phi(&self, fim: Mat<f64>) -> Mat<f64> {
        let s = fim.partial_piv_lu().solve(&self.design_t);
        &self.design * s
    }
}

impl<const D: usize> NLPFunctionTarget for DMatrixMean<D> {
    #[inline(always)]
    fn val(&self, x: &Mat<f64>) -> f64 {
        let (_, det) = self.fim_det(x);
        -det.ln()
    }

    #[inline(always)]
    fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
        let (fim, det) = self.fim_det(x);
        let phi = self.phi(fim);
        let val = -det.ln();
        let mut grad = Mat::<f64>::zeros(phi.nrows(), 1);
        for row in 0..grad.nrows() {
            let v = phi[(row, row)];
            grad[(row, 0)] = -v;
        }
        (val, grad)
    }

    #[inline(always)]
    fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
        let (fim, det) = self.fim_det(x);
        let phi = self.phi(fim);
        let val = -det.ln();
        let mut grad = Mat::<f64>::zeros(phi.nrows(), 1);
        for row in 0..grad.nrows() {
            let v = phi[(row, row)];
            grad[(row, 0)] = -v;
        }
        let mut hes = Mat::<f64>::zeros(phi.nrows(), phi.ncols());
        for col in 0..phi.ncols() {
            for row in col..phi.nrows() {
                let v = phi[(row, col)];
                hes[(row, col)] = v * v;
            }
        }

        (val, grad, hes)
    }
}

pub struct DDispersionFunction<const D: usize> {
    linear_model: Arc<LinearModel<D>>,
    fim_lu: PartialPivLu<f64>,
}

impl<const D: usize> DDispersionFunction<D> {
    pub fn new(
        linear_model: Arc<LinearModel<D>>,
        weights: Mat<f64>,
        supp: Arc<MatrixDRows<D>>,
    ) -> Self {
        let fim = linear_model.fim(&supp, &weights);
        let fim_lu = fim.partial_piv_lu();
        Self {
            linear_model,
            fim_lu,
        }
    }

    fn pre_calculations(&self, x: &SVector<f64, D>) -> (Mat<f64>, Mat<f64>) {
        let feature_vec = self.linear_model.feature_vec(x);
        let inv_fim_design = self.fim_lu.solve(&feature_vec);
        (feature_vec, inv_fim_design)
    }
}

impl<const D: usize> NLPFunctionTarget for DDispersionFunction<D> {
    fn val(&self, x: &Mat<f64>) -> f64 {
        let x_svector = x.into_svector();
        let (feature_vec, inv_fim_design) = self.pre_calculations(&x_svector);
        let val = -feature_vec.col(0).transpose() * inv_fim_design.col(0)
            + self.linear_model.features.len() as f64;
        val
    }

    fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
        let x_svector = x.into_svector();
        let (feature_vec, inv_fim_design) = self.pre_calculations(&x_svector);
        let jac_t = self.linear_model.jac_t(&x_svector);
        let val = -feature_vec.col(0).transpose() * inv_fim_design.col(0)
            + self.linear_model.features.len() as f64;
        let grad = -&jac_t * inv_fim_design;
        (val, grad)
    }

    fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
        let x_svector = x.into_svector();
        let (feature_vec, inv_fim_design) = self.pre_calculations(&x_svector);
        let jac_t = self.linear_model.jac_t(&x_svector);
        let val = -feature_vec.col(0).transpose() * inv_fim_design.col(0)
            + self.linear_model.features.len() as f64;
        let grad = -&jac_t * &inv_fim_design;
        let mut hes = -&jac_t * self.fim_lu.solve(jac_t.transpose());
        self.linear_model
            .feature_vec_hessian(&x_svector)
            .iter()
            .enumerate()
            .for_each(|(idx, feat_hes)| {
                hes -= feat_hes * inv_fim_design[(idx, 0)];
            });
        (val, grad, hes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_nlp_target_consistency, Feature, FeatureFunction, Result};
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
        let monom = Monomial { i: 1, j: 1 };
        LinearModel::new(vec![Arc::new(monom)])
    }

    #[test]
    fn d_crit_weight_consistency() -> Result<()> {
        let linear_model: Arc<_> = get_linear_model().into();
        let supp: Arc<_> = MatrixDRows::from_vec(vec![1., 2., 3., 4.]).into();
        let d_opt = DOptimality::new(linear_model);
        let weights = Mat::<f64>::ones(2, 1);
        assert!((d_opt.measure(&weights, supp) - 148.) < 1e-8);
        Ok(())
    }

    #[test]
    fn d_crit_supp_consistency() -> Result<()> {
        let linear_model: Arc<_> = get_linear_model().into();
        let supp: Arc<_> = MatrixDRows::from_vec(vec![1., 2., 3., 4.]).into();
        let weights = Mat::<f64>::ones(2, 1);

        let dcrit = DDispersionFunction::new(linear_model, weights, supp);
        let x = Mat::<f64>::ones(2, 1);
        assert_nlp_target_consistency!(dcrit, &x);

        Ok(())
    }
}
