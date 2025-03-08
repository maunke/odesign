mod a_opt;
mod c_opt;
mod d_opt;
use crate::{MatrixDRows, NLPFunctionTarget};
pub use a_opt::AOptimality;
pub use c_opt::COptimality;
pub use d_opt::DOptimality;
use faer::Mat;
use rayon::prelude::*;
use std::sync::Arc;

/// Vector of optimalities
pub type Optimalities<const D: usize> = Vec<Arc<dyn Optimality<D> + Send + Sync>>;

/// Defines a list of optimality measures and their weights.
#[derive(Default)]
pub struct OptimalityMeasures {
    measures: Vec<Arc<dyn NLPFunctionTarget + Sync + Send>>,
    weights: Vec<f64>,
}

impl OptimalityMeasures {
    /// Creates new struct with empty list of measures and weights.
    pub fn new() -> Self {
        Self::default()
    }
    /// Adds an optimality measure and its weight.
    pub fn push(&mut self, measure: Arc<dyn NLPFunctionTarget + Sync + Send>, weight: f64) {
        self.measures.push(measure);
        self.weights.push(weight);
    }
}

impl NLPFunctionTarget for OptimalityMeasures {
    fn val(&self, x: &Mat<f64>) -> f64 {
        match self.measures.len() {
            0 => 0.,
            1 => self.measures[0].val(x),
            2..3 => self
                .measures
                .iter()
                .zip(&self.weights)
                .map(|(m, w)| w * m.val(x))
                .sum(),
            3.. => self
                .measures
                .par_iter()
                .zip(&self.weights)
                .map(|(m, w)| w * m.val(x))
                .sum(),
        }
    }

    fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
        match self.measures.len() {
            0 => (0.0, Mat::zeros(x.nrows(), 1)),
            1 => self.measures[0].val_grad(x),
            2..3 => self
                .measures
                .iter()
                .zip(&self.weights)
                .map(|(m, w)| {
                    let val_grad = m.val_grad(x);
                    (w * val_grad.0, *w * val_grad.1)
                })
                .reduce(|(mut sum_val, mut sum_grad), (val, grad)| {
                    sum_val += val;
                    sum_grad += grad;
                    (sum_val, sum_grad)
                })
                .unwrap(),
            3.. => self
                .measures
                .par_iter()
                .zip(&self.weights)
                .map(|(m, w)| {
                    let val_grad = m.val_grad(x);
                    (w * val_grad.0, *w * val_grad.1)
                })
                .reduce(
                    || (0.0, Mat::zeros(x.nrows(), 1)),
                    |(mut sum_val, mut sum_grad), (val, grad)| {
                        sum_val += val;
                        sum_grad += grad;
                        (sum_val, sum_grad)
                    },
                ),
        }
    }
    fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
        match self.measures.len() {
            0 => (
                0.0,
                Mat::zeros(x.nrows(), 1),
                Mat::zeros(x.nrows(), x.nrows()),
            ),
            1 => self.measures[0].val_grad_hes(x),
            2..3 => self
                .measures
                .iter()
                .zip(&self.weights)
                .map(|(m, w)| {
                    let val_grad_hes = m.val_grad_hes(x);
                    (w * val_grad_hes.0, *w * val_grad_hes.1, *w * val_grad_hes.2)
                })
                .reduce(
                    |(mut sum_val, mut sum_grad, mut sum_hes), (val, grad, hes)| {
                        sum_val += val;
                        sum_grad += grad;
                        sum_hes += hes;
                        (sum_val, sum_grad, sum_hes)
                    },
                )
                .unwrap(),
            3.. => self
                .measures
                .par_iter()
                .zip(&self.weights)
                .map(|(m, w)| {
                    let val_grad_hes = m.val_grad_hes(x);
                    (w * val_grad_hes.0, *w * val_grad_hes.1, *w * val_grad_hes.2)
                })
                .reduce(
                    || {
                        (
                            0.0,
                            Mat::zeros(x.nrows(), 1),
                            Mat::zeros(x.nrows(), x.nrows()),
                        )
                    },
                    |(mut sum_val, mut sum_grad, mut sum_hes), (val, grad, hes)| {
                        sum_val += val;
                        sum_grad += grad;
                        sum_hes += hes;
                        (sum_val, sum_grad, sum_hes)
                    },
                ),
        }
    }
}

/// Defines an optimality by providing the matrix mean, despersion function and a measure function.
pub trait Optimality<const D: usize> {
    /// A twice continuously differentiable matrix mean function of the optimality.
    fn matrix_mean(&self, supp: Arc<MatrixDRows<D>>) -> Arc<dyn NLPFunctionTarget + Send + Sync>;
    /// A twice continuously differentiable dispersion function of the optimality.
    fn dispersion_function(
        &self,
        supp: Arc<MatrixDRows<D>>,
        weights: Mat<f64>,
        x_id: usize,
    ) -> Arc<dyn NLPFunctionTarget + Send + Sync>;
    /// Returns the optimality measure value.
    fn measure(&self, weights: &Mat<f64>, supp: Arc<MatrixDRows<D>>) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Result, assert_nlp_target_consistency};

    struct DesignTest {}

    impl NLPFunctionTarget for DesignTest {
        fn val(&self, _x: &Mat<f64>) -> f64 {
            1.
        }
        fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
            (self.val(x), Mat::ones(x.nrows(), 1))
        }

        fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
            let val_grad = self.val_grad(x);
            (val_grad.0, val_grad.1, Mat::ones(x.nrows(), x.nrows()))
        }
    }

    #[test]
    fn design_measures() -> Result<()> {
        let dt = Arc::new(DesignTest {});
        for no_measures in 0..42 {
            let mut dms = OptimalityMeasures::new();
            (0..no_measures).for_each(|w| dms.push(dt.clone(), w as f64 + 1.));
            let x = Mat::ones(1, 1);
            let vgh = dms.val_grad_hes(&x);
            let no_measures_f = no_measures as f64;
            let factor = no_measures_f * (no_measures_f + 1.) / 2.;
            let vgh_val = factor;
            let vgh_grad = factor * Mat::<f64>::ones(1, 1);
            let vgh_hes = factor * Mat::<f64>::ones(1, 1);
            assert_eq!(vgh.0, vgh_val);
            assert_eq!(vgh.1, vgh_grad);
            assert_eq!(vgh.2, vgh_hes);
            assert_nlp_target_consistency!(dms, &x);
        }
        Ok(())
    }
}
