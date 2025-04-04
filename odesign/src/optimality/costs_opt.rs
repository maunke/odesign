use crate::{MatrixDRows, NLPFunctionTarget, Optimality, utils::MatrixFind};
use faer::Mat;
use std::sync::Arc;

#[cfg_attr(doc, katexit::katexit)]
/// The Costs-Optimality is defined as the reciproce exponential negative sum of quadratic
/// penalized weights of support vectors that are not measured yet.
///
/// $$
/// \text{Costs}_{opt} := \exp{(-\sum_w \begin{cases}
/// (1 + alpha * w)^2 & \text{if } x \notin \xi_m \text{, }\\
/// 0 & \text{else} \\
/// \end{cases})}
/// $$
///
/// with the measured support-vectors $x \in \xi_m$.
///
#[derive(Clone)]
pub struct CostsOptimality<const D: usize> {
    measurements: Arc<MatrixDRows<D>>,
    alpha: f64,
}

impl<const D: usize> CostsOptimality<D> {
    /// Instantizes [CostsOptimality]
    pub fn new(measurements: Arc<MatrixDRows<D>>, alpha: f64) -> Self {
        Self {
            measurements,
            alpha,
        }
    }
}

impl<const D: usize> Optimality<D> for CostsOptimality<D> {
    fn measure(&self, weights: &Mat<f64>, supp: Arc<MatrixDRows<D>>) -> f64 {
        let m_mean = self.matrix_mean(supp);
        let val = m_mean.val(weights);
        (-val).exp()
    }

    fn matrix_mean(&self, supp: Arc<MatrixDRows<D>>) -> Arc<dyn NLPFunctionTarget + Send + Sync> {
        let supp_measured_indices = supp.find_col_indices(&self.measurements);
        let supp_not_measured_indices: Vec<usize> = (0..supp.ncols())
            .filter(|idx| !supp_measured_indices.contains(idx))
            .collect();
        Arc::new(MatrixMeans::new(supp_not_measured_indices, self.alpha))
    }

    fn dispersion_function(
        &self,
        _supp: Arc<MatrixDRows<D>>,
        _weights: Mat<f64>,
        _x_id: usize,
    ) -> Arc<dyn NLPFunctionTarget + Send + Sync> {
        Arc::new(DispersionFunction::new(D))
    }
}

struct MatrixMeans {
    supp_not_measured_indices: Vec<usize>,
    alpha: f64,
}

impl MatrixMeans {
    fn new(supp_not_measured_indices: Vec<usize>, alpha: f64) -> Self {
        Self {
            supp_not_measured_indices,
            alpha,
        }
    }
}

impl NLPFunctionTarget for MatrixMeans {
    fn val(&self, x: &Mat<f64>) -> f64 {
        let mut val = 0.;
        for &idx in &self.supp_not_measured_indices {
            let weight = x[(idx, 0)];
            let phi = 1. + self.alpha * weight;
            val += phi.powi(2);
        }
        val
    }

    fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
        let mut val = 0.;
        let mut grad = Mat::<f64>::zeros(x.nrows(), 1);
        for &idx in &self.supp_not_measured_indices {
            let weight = x[(idx, 0)];
            let phi = 1. + self.alpha * weight;
            val += phi.powi(2);
            grad[(idx, 0)] = 2. * self.alpha * phi;
        }
        (val, grad)
    }

    fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
        let mut val = 0.;
        let mut grad = Mat::<f64>::zeros(x.nrows(), 1);
        let mut hes = Mat::<f64>::zeros(x.nrows(), x.nrows());
        for &idx in &self.supp_not_measured_indices {
            let weight = x[(idx, 0)];
            let phi = 1. + self.alpha * weight;
            val += phi.powi(2);
            grad[(idx, 0)] = 2. * self.alpha * phi;
            hes[(idx, idx)] = 2. * self.alpha.powi(2);
        }
        (val, grad, hes)
    }
}

struct DispersionFunction {
    grad: Mat<f64>,
    hes: Mat<f64>,
}

impl DispersionFunction {
    fn new(dim: usize) -> Self {
        let grad = Mat::<f64>::zeros(dim, 1);
        let hes = Mat::<f64>::zeros(dim, dim);
        Self { grad, hes }
    }
}

impl NLPFunctionTarget for DispersionFunction {
    fn val(&self, _x: &Mat<f64>) -> f64 {
        0.
    }

    fn val_grad(&self, _x: &Mat<f64>) -> (f64, Mat<f64>) {
        (0., self.grad.clone())
    }

    fn val_grad_hes(&self, _x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
        (0., self.grad.clone(), self.hes.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MatrixUnion;
    use crate::Result;
    use crate::assert_nlp_target_consistency;

    #[test]
    fn costs_matrix_mean_consistency() -> Result<()> {
        let init: Arc<_> = MatrixDRows::from_vec(vec![1., 2., 3., 4.]).into();
        let measurements: Arc<_> = MatrixDRows::from_vec(vec![1., 2.]).into();
        let supp = init.union(&measurements);
        let alpha = 1.;
        let crit = CostsOptimality::<2>::new(measurements, alpha);
        let weights = Mat::<f64>::ones(2, 1);
        let mean = crit.matrix_mean(supp.clone().into());
        let x = Mat::<f64>::ones(2, 1);
        let disp = crit.dispersion_function(supp.into(), x.clone(), 0);
        assert_nlp_target_consistency!(mean, &weights);
        assert_nlp_target_consistency!(disp, &x);
        Ok(())
    }
}
