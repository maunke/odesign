use faer::Mat;
use nalgebra::{Const, Dyn, Matrix, SVector, VecStorage};

use crate::{Error, Result};

/// Weight type with a value between 0.0 and 1.0.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Weight(f64);

/// Represents a range of weights with a minimum and maximum weight.
#[derive(Clone)]
pub struct WeightRange {
    min: Weight,
    max: Weight,
}

impl WeightRange {
    /// Creates a new `WeightRange` with the given minimum and maximum weights.
    pub fn new(min: Weight, max: Weight) -> Result<Self> {
        if max < min {
            return Err(Error::MaxWeightSmallerThanMin { max, min });
        }
        Ok(Self { min, max })
    }
}

/// Represents a function that maps a position to a weight.
pub trait WeightsFunction {
    /// Returns the weight at the given position.
    fn weight(&self, pos: usize, max_pos: usize) -> Weight;
}

impl WeightsFunction for WeightRange {
    fn weight(&self, pos: usize, max_pos: usize) -> Weight {
        self.min.lerp(
            self.max,
            Weight::try_from(pos as f64 / (max_pos as f64 - 1.))
                .expect("pos < max_pos by construction"),
        )
    }
}

impl Weight {
    /// Returns the raw value of the weight.
    pub fn get(&self) -> f64 {
        self.0
    }

    /// Returns the linear interpolation between two weights as weight.
    pub fn lerp(self, to: Weight, t: Weight) -> Self {
        Self(self.0 + (to.0 - self.0) * t.0)
    }
}

impl TryFrom<f64> for Weight {
    type Error = Error;

    fn try_from(value: f64) -> std::result::Result<Self, Self::Error> {
        if value.is_nan() {
            return Err(Error::WeightIsNaN);
        } else if !(0.0..=1.0).contains(&value) {
            return Err(Error::WeightOutOfRange { value });
        }
        Ok(Self(value))
    }
}

impl From<Weight> for f64 {
    fn from(value: Weight) -> Self {
        value.0
    }
}

impl std::fmt::Display for Weight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Matrix with D rows
pub type MatrixDRows<const D: usize> = Matrix<f64, Const<D>, Dyn, VecStorage<f64, Const<D>, Dyn>>;

/// Deduplication of columns
pub trait MatrixUnion<const D: usize> {
    /// Returns a matrix with deduplicated columns of self and matrix b
    fn union(&self, b: &MatrixDRows<D>) -> MatrixDRows<D>;
}

impl<const D: usize> MatrixUnion<D> for MatrixDRows<D> {
    fn union(&self, b: &MatrixDRows<D>) -> MatrixDRows<D> {
        let new_cols: Vec<_> = b
            .column_iter()
            .filter(|&col| {
                for x in self.column_iter() {
                    if (x - col).norm() < 1e-8 {
                        return false;
                    }
                }
                true
            })
            .collect();
        let mut cols: Vec<_> = self.column_iter().collect();
        cols.extend(new_cols);
        MatrixDRows::<D>::from_columns(&cols)
    }
}

pub trait MatrixFind<const D: usize> {
    fn find_col_indices(&self, mat: &MatrixDRows<D>) -> Vec<usize>;
}

impl<const D: usize> MatrixFind<D> for MatrixDRows<D> {
    fn find_col_indices(&self, mat: &MatrixDRows<D>) -> Vec<usize> {
        self.column_iter()
            .enumerate()
            .filter(|(_, col)| {
                for m in mat.column_iter() {
                    if (col - m).norm() < 1e-8 {
                        return true;
                    }
                }
                false
            })
            .map(|(idx, _)| idx)
            .collect()
    }
}

/// Convert type to svector with D entries
pub trait IntoSVector<const D: usize> {
    /// Returns svector of fixed size D
    fn into_svector(self) -> SVector<f64, D>;
}

impl<const D: usize> IntoSVector<D> for &Mat<f64> {
    fn into_svector(self) -> SVector<f64, D> {
        let x_slice = &self.col_as_slice(0)[0..D];
        SVector::from_row_slice(x_slice)
    }
}

fn subsets_generator(subset: Vec<usize>, pos: usize, k: usize, s: usize) -> Vec<Vec<usize>> {
    if subset.len() == k {
        vec![subset]
    } else if pos < s {
        let mut subsets = vec![];
        for idx in pos..s {
            let mut s_subset = subset.clone();
            s_subset.push(idx);
            subsets.extend(subsets_generator(s_subset, idx + 1, k, s));
        }
        subsets
    } else {
        vec![]
    }
}

/// Returns the set of index subsets with length k out of a set of the size s
pub fn subsets_len_k_in_s(k: usize, s: usize) -> Vec<Vec<usize>> {
    if s < k || (k == s && s == 0) || k == 0 {
        return vec![];
    }
    subsets_generator(vec![], 0, k, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Result;
    use faer::mat;

    /// Binomial coefficient
    fn binom_coeff(n: usize, mut k: usize) -> usize {
        let mut c = 1;
        // use symmetry
        if k > n - k {
            k = n - k;
        }
        for i in 0..k {
            c *= n - i;
            c /= i + 1;
        }
        c
    }

    #[test]
    fn test_into_svector() -> Result<()> {
        let mat = mat![[1.], [1.], [1.]];
        let _svector: SVector<f64, 3> = mat.into_svector();
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_into_svector_panic() {
        let mat = mat![[1.], [1.], [1.]];
        let _svector: SVector<f64, 4> = mat.into_svector();
    }

    #[test]
    fn test_matrix_union() {
        let mat_a: MatrixDRows<2> = MatrixDRows::from_vec(vec![0., 0., 0., 1., 1., 0., 1., 1.]);
        let mat_b: MatrixDRows<2> = MatrixDRows::from_vec(vec![0., 1., 1., 0., 1., 1., 2., 2.]);
        let mat_union = mat_a.union(&mat_b);
        let mat_assert: MatrixDRows<2> =
            MatrixDRows::from_vec(vec![0., 0., 0., 1., 1., 0., 1., 1., 2., 2.]);
        assert_eq!(mat_union, mat_assert);
    }

    #[test]
    fn test_find_col_indices() {
        let mat_a: MatrixDRows<2> = MatrixDRows::from_vec(vec![0., 0., 0., 1., 1., 0., 1., 1.]);
        let mat_b: MatrixDRows<2> = MatrixDRows::from_vec(vec![0., 1., 1., 0., 1., 1., 2., 2.]);
        let vec_find_by_indices = mat_a.find_col_indices(&mat_b);
        let vec_assert = vec![1, 2, 3];
        assert_eq!(vec_find_by_indices, vec_assert);
    }

    #[test]
    fn test_binom_coeff() {
        // (n, k, binom_coeff)
        let lookup: Vec<(usize, usize, usize)> = vec![
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 1),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 2),
            (3, 0, 1),
            (3, 1, 3),
            (3, 2, 3),
            (4, 0, 1),
            (4, 1, 4),
            (4, 2, 6),
            (14, 3, 364),
        ];

        for &l in lookup.iter() {
            assert_eq!(binom_coeff(l.0, l.1), l.2)
        }
    }

    #[test]
    fn test_subsets_len_k_in_s() {
        let tests: Vec<(usize, usize, Vec<Vec<usize>>)> = vec![
            (1, 1, vec![vec![0]]),
            (1, 2, vec![vec![0], vec![1]]),
            (2, 3, vec![vec![0, 1], vec![0, 2], vec![1, 2]]),
            (
                2,
                4,
                vec![
                    vec![0, 1],
                    vec![0, 2],
                    vec![0, 3],
                    vec![1, 2],
                    vec![1, 3],
                    vec![2, 3],
                ],
            ),
        ];

        for t in tests {
            let s = subsets_len_k_in_s(t.0, t.1);
            assert_eq!(s.len(), t.2.len());
            assert_eq!(s.len(), binom_coeff(t.1, t.0));
            for (a, b) in s.into_iter().zip(t.2) {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn test_weight() {
        let error_values = [2.0, f64::MIN, f64::MAX];
        for value in error_values {
            assert_eq!(
                Weight::try_from(value).err(),
                Some(Error::WeightOutOfRange { value })
            );
        }
        assert!(Weight::try_from(0.).is_ok());
        assert!(Weight::try_from(1.).is_ok());
        assert!(Weight::try_from(0.5).is_ok());
        assert!(Weight::try_from(-1.).is_err());
        assert!(Weight::try_from(2.).is_err());

        // NaN should return WeightIsNaN error.
        assert_eq!(Weight::try_from(f64::NAN).err(), Some(Error::WeightIsNaN));
    }

    #[test]
    fn test_weight_lerp() -> Result<()> {
        let w1 = Weight::try_from(0.5)?;
        let w2 = Weight::try_from(1.0)?;
        let t = Weight::try_from(0.5)?;
        let result = w1.lerp(w2, t);
        assert_eq!(result, Weight::try_from(0.75)?);
        Ok(())
    }
}
