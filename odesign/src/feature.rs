use crate::{Error, Result};
use nalgebra::{SMatrix, SVector};
use num_dual::DualNum;
use std::collections::HashSet;
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
#[derive(Clone, Default)]
pub struct FeatureSet<const D: usize> {
    // Vectoring storing features.
    features: Vec<Arc<dyn Feature<D> + Send + Sync>>,
}

impl<const D: usize> FeatureSet<D> {
    /// Create empty feature set.
    pub fn new() -> FeatureSet<D> {
        Self::default()
    }

    /// Add feature to feature set.
    pub fn push<F: Feature<D> + Send + Sync + 'static>(&mut self, feature: F) {
        self.features.push(Arc::new(feature))
    }

    /// Returns an iterator over the features.
    pub fn iter(&self) -> std::slice::Iter<'_, Arc<dyn Feature<D> + Send + Sync>> {
        self.features.iter()
    }

    /// Returns the number of features.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Returns a subset of the features selected by the given indices.
    ///
    /// Each index must be unique and in range; otherwise returns
    /// [`Error::DuplicateIndex`] or [`Error::IndexOutOfBounds`]. These checks also
    /// guarantee termination on unbounded iterators (e.g. `1..`), since after at
    /// most `len + 1` items the iterator must yield either a duplicate or an
    /// out-of-bounds index.
    pub fn subset<I: IntoIterator<Item = usize>>(&self, indices: I) -> Result<FeatureSet<D>> {
        let len = self.features.len();
        let mut dedup_book = HashSet::with_capacity(len);
        let features = indices
            .into_iter()
            .map(|index| {
                if !dedup_book.insert(index) {
                    return Err(Error::DuplicateIndex { index });
                }
                self.features
                    .get(index)
                    .cloned()
                    .ok_or(Error::IndexOutOfBounds { index, len })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(FeatureSet { features })
    }
}

impl<const D: usize, F> From<Vec<F>> for FeatureSet<D>
where
    F: Feature<D> + Send + Sync + 'static,
{
    fn from(features: Vec<F>) -> Self {
        Self {
            features: features
                .into_iter()
                .map(|f| Arc::new(f) as Arc<dyn Feature<D> + Send + Sync>)
                .collect(),
        }
    }
}

impl<const D: usize> IntoIterator for FeatureSet<D> {
    type Item = Arc<dyn Feature<D> + Send + Sync>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.features.into_iter()
    }
}

impl<const D: usize> FromIterator<Arc<dyn Feature<D> + Send + Sync>> for FeatureSet<D> {
    fn from_iter<I: IntoIterator<Item = Arc<dyn Feature<D> + Send + Sync>>>(iter: I) -> Self {
        Self {
            features: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Result;
    use nalgebra::{Matrix2, Vector2};
    use num_dual::DualNum;
    use odesign_derive::Feature;
    use rand::RngExt;

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
        let feature = Monomial { i: 1, j: 2 };
        fs.push(feature);
        assert_eq!(fs.features.len(), 1);
        Ok(())
    }

    #[test]
    fn subset_rejects_duplicates_and_oob() -> Result<()> {
        let fs: FeatureSet<2> = vec![
            Monomial { i: 0, j: 0 },
            Monomial { i: 1, j: 0 },
            Monomial { i: 0, j: 1 },
        ]
        .into();

        assert!(matches!(
            fs.subset([0, 0]),
            Err(crate::Error::DuplicateIndex { index: 0 })
        ));
        assert!(matches!(
            fs.subset([3]),
            Err(crate::Error::IndexOutOfBounds { index: 3, len: 3 })
        ));
        // Unbounded range terminates: yields 1, 2, 3 -> 3 is OOB.
        assert!(matches!(
            fs.subset(1..),
            Err(crate::Error::IndexOutOfBounds { index: 3, len: 3 })
        ));
        assert_eq!(fs.subset(0..3)?.len(), 3);
        Ok(())
    }
}
