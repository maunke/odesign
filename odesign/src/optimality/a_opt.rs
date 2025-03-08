use crate::{IntoSVector, LinearModel, MatrixDRows, NLPFunctionTarget, Optimality};
use faer::{Mat, Side, linalg::solvers::DenseSolveCore};
use std::sync::Arc;

#[cfg_attr(doc, katexit::katexit)]
/// A-Optimality is defined as reciproce trace of the inverse of the
/// fisher-information matrix.
///
/// $$ A_{opt} := \frac{n}{\text{tr}\mathcal{M}^{-1}} $$
///
/// with fisher-information matrix $\mathcal{M}$ and number of coefficients $n$.
///
#[derive(Clone)]
pub struct AOptimality<const D: usize> {
    linear_model: Arc<LinearModel<D>>,
}

impl<const D: usize> AOptimality<D> {
    /// Instantizes [AOptimality]
    pub fn new(linear_model: Arc<LinearModel<D>>) -> Self {
        Self { linear_model }
    }
}

impl<const D: usize> Optimality<D> for AOptimality<D> {
    fn matrix_mean(&self, supp: Arc<MatrixDRows<D>>) -> Arc<dyn NLPFunctionTarget + Send + Sync> {
        Arc::new(MatrixMean::new(self.linear_model.clone(), supp))
    }
    fn dispersion_function(
        &self,
        supp: Arc<MatrixDRows<D>>,
        weights: Mat<f64>,
        _x_id: usize,
    ) -> Arc<dyn NLPFunctionTarget + Send + Sync> {
        Arc::new(DispersionFunction::new(
            self.linear_model.clone(),
            weights,
            supp,
        ))
    }
    fn measure(&self, weights: &Mat<f64>, supp: Arc<MatrixDRows<D>>) -> f64 {
        let m_mean = self.matrix_mean(supp);
        let val = m_mean.val(weights);
        (self.linear_model.features.len() as f64) / val.exp()
    }
}

struct MatrixMean<const D: usize> {
    linear_model: Arc<LinearModel<D>>,
    design_t: Mat<f64>,
    design: Mat<f64>,
}

impl<const D: usize> MatrixMean<D> {
    pub fn new(linear_model: Arc<LinearModel<D>>, supp: Arc<MatrixDRows<D>>) -> Self {
        let design_t = linear_model.design_t(&supp);
        let design = design_t.transpose().to_owned();
        Self {
            linear_model,
            design_t,
            design,
        }
    }

    #[inline(always)]
    fn fim_inv_trace(&self, x: &Mat<f64>) -> (Mat<f64>, f64) {
        let fim = self.linear_model.fim_from_design_t(&self.design_t, x);
        let fim_inv = fim.lblt(Side::Lower).inverse();
        let trace = fim_inv.diagonal().column_vector().sum();
        (fim_inv, trace)
    }

    #[inline(always)]
    fn z(&self, fim_inv: &Mat<f64>) -> Mat<f64> {
        &self.design * (fim_inv * fim_inv) * &self.design_t
    }

    #[inline(always)]
    fn z_one_and_two(&self, fim_inv: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
        let tmp = fim_inv * &self.design_t;
        (&self.design * &tmp, &self.design * fim_inv * &tmp)
    }
}

impl<const D: usize> NLPFunctionTarget for MatrixMean<D> {
    #[inline(always)]
    fn val(&self, x: &Mat<f64>) -> f64 {
        let (_, trace) = self.fim_inv_trace(x);
        trace.ln()
    }

    #[inline(always)]
    fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
        let (fim_inv, trace) = self.fim_inv_trace(x);
        let z = self.z(&fim_inv);
        let val = trace.ln();
        let mut grad = Mat::<f64>::zeros(z.nrows(), 1);
        for row in 0..grad.nrows() {
            let v = z[(row, row)];
            grad[(row, 0)] = v;
        }
        (val, grad)
    }

    #[inline(always)]
    fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
        let (fim_inv, trace) = self.fim_inv_trace(x);
        let (z_one, z_two) = self.z_one_and_two(&fim_inv);
        let val = trace.ln();
        let mut grad = Mat::<f64>::zeros(z_two.nrows(), 1);
        for row in 0..grad.nrows() {
            let v = z_two[(row, row)];
            grad[(row, 0)] = v;
        }
        let mut hes = Mat::<f64>::zeros(z_two.nrows(), z_two.ncols());
        for col in 0..z_two.ncols() {
            for row in col..z_two.nrows() {
                let v1 = z_one[(row, col)];
                let v2 = z_two[(row, col)];
                hes[(row, col)] = -2.0 * v1 * v2;
            }
        }

        (val, grad, hes)
    }
}

pub struct DispersionFunction<const D: usize> {
    linear_model: Arc<LinearModel<D>>,
    fim_inv_two: Mat<f64>,
    trace_fim_inv: f64,
}

impl<const D: usize> DispersionFunction<D> {
    pub fn new(
        linear_model: Arc<LinearModel<D>>,
        weights: Mat<f64>,
        supp: Arc<MatrixDRows<D>>,
    ) -> Self {
        let fim = linear_model.fim(&supp, &weights);
        let fim_l = fim.lblt(Side::Lower);
        let fim_inv = fim_l.inverse();
        let fim_inv_two = &fim_inv * &fim_inv;
        let trace_fim_inv = fim_inv.diagonal().column_vector().sum();
        Self {
            linear_model,
            fim_inv_two,
            trace_fim_inv,
        }
    }
}

impl<const D: usize> NLPFunctionTarget for DispersionFunction<D> {
    fn val(&self, x: &Mat<f64>) -> f64 {
        let x_svector = x.into_svector();
        let feature_vec = self.linear_model.feature_vec(&x_svector);
        -feature_vec.col(0).transpose() * &self.fim_inv_two * feature_vec.col(0)
            + self.trace_fim_inv
    }

    fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
        let x_svector = x.into_svector();
        let feature_vec = self.linear_model.feature_vec(&x_svector);
        let val = -feature_vec.col(0).transpose() * &self.fim_inv_two * feature_vec.col(0)
            + self.trace_fim_inv;
        let jac_t = self.linear_model.jac_t(&x_svector);
        let grad = -&jac_t * &self.fim_inv_two * feature_vec;
        (val, grad)
    }

    fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
        let x_svector = x.into_svector();
        let feature_vec = self.linear_model.feature_vec(&x_svector);
        let val = -feature_vec.col(0).transpose() * &self.fim_inv_two * feature_vec.col(0)
            + self.trace_fim_inv;
        let jac_t = self.linear_model.jac_t(&x_svector);
        let inv_fim_design = &self.fim_inv_two * feature_vec;
        let grad = -&jac_t * &inv_fim_design;
        let mut hes = -&jac_t * &self.fim_inv_two * jac_t.transpose();
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
    use crate::{Feature, FeatureFunction, FeatureSet, Result, assert_nlp_target_consistency};
    use nalgebra::SVector;
    use num_dual::DualNum;

    #[derive(Feature)]
    #[dimension = 1]
    struct Monomial {
        i: i32,
    }

    impl FeatureFunction<1> for Monomial {
        fn f<D: DualNum<f64>>(&self, x: &SVector<D, 1>) -> D {
            x[0].powi(self.i)
        }
    }

    fn get_linear_model() -> LinearModel<1> {
        let mut fs = FeatureSet::new();
        for i in 0..3 {
            let c: Arc<_> = Monomial { i }.into();
            fs.push(c);
        }
        LinearModel::new(fs.features)
    }

    #[test]
    fn test_a_d_crit_supp_consistency() -> Result<()> {
        let linear_model: Arc<_> = get_linear_model().into();
        let supp: Arc<_> = MatrixDRows::from_vec(vec![1., 2., 3., 4.]).into();
        let weights = Mat::<f64>::ones(4, 1);

        let dcrit = DispersionFunction::new(linear_model, weights, supp);
        let x = Mat::<f64>::ones(1, 1);
        assert_nlp_target_consistency!(dcrit, &x);

        Ok(())
    }
}
