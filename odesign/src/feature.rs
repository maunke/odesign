use nalgebra::{SMatrix, SVector};
use num_dual::DualNum;
use std::sync::Arc;

/// Required value function for [Feature] derive.
pub trait FeatureFunction<const N: usize> {
    /// Defines value function of a feature.
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, N>) -> D;
}

/// Defines the value, gradient and hessian functions of a feature.
pub trait Feature<const D: usize> {
    /// Value function.
    fn val(&self, x: &SVector<f64, D>) -> f64;
    /// Value and gradient function.
    fn val_grad(&self, x: &SVector<f64, D>) -> (f64, SVector<f64, D>);
    /// Value, gradient and hessian function.
    fn val_grad_hes(&self, x: &SVector<f64, D>) -> (f64, SVector<f64, D>, SMatrix<f64, D, D>);
}

/// Set of features.
#[derive(Default)]
pub struct FeatureSet<const D: usize> {
    /// Vectoring storing features.
    pub features: Vec<Arc<dyn Feature<D> + Send + Sync>>,
}

impl<const D: usize> FeatureSet<D> {
    /// Create empty feature set.
    pub fn new() -> FeatureSet<D> {
        Self::default()
    }

    /// Add feature to feature set.
    pub fn push(&mut self, feature: Arc<dyn Feature<D> + Send + Sync>) {
        self.features.push(feature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Result;
    use nalgebra::{Matrix2, Vector2};
    use num_dual::DualNum;
    use odesign_derive::Feature;
    use rand::Rng;

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

    fn monom_hessian(i: i32, j: i32, x: &Vector2<f64>) -> Matrix2<f64> {
        let hi = i as f64;
        let hj = j as f64;
        Matrix2::new(
            hi * (hi - 1.) * x[0].powi(i - 2) * x[1].powi(j),
            hi * hj * x[0].powi(i - 1) * x[1].powi(j - 1),
            hi * hj * x[0].powi(i - 1) * x[1].powi(j - 1),
            hj * (hj - 1.) * x[0].powi(i) * x[1].powi(j - 2),
        )
    }

    #[test]
    fn feature_derive() -> Result<()> {
        let mut rng = rand::rng();
        for j in 0..10 {
            for i in 0..42 {
                let p = Monomial { i, j };
                let x = Vector2::new(2. * rng.random::<f64>(), 3. * rng.random::<f64>());
                let hessian = p.val_grad_hes(&x).2;
                let hessian_rslt = monom_hessian(i, j, &x);
                assert!(hessian.relative_eq(&hessian_rslt, EQ_EPS, EQ_MAX_REL));
            }
        }
        Ok(())
    }

    #[test]
    fn feature_set() -> Result<()> {
        let mut fs = FeatureSet::new();
        let feature: Arc<_> = Monomial { i: 1, j: 2 }.into();
        fs.push(feature);
        assert_eq!(fs.features.len(), 1);
        Ok(())
    }
}
