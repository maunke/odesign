use faer::Mat;
use nalgebra::{Const, Dyn, Matrix, SVector, VecStorage};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Result;
    use faer::mat;

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
}
