use faer::Mat;
use nalgebra::{Const, Dyn, Matrix, SVector, VecStorage};

/// Matrix with D rows
pub type MatrixDRows<const D: usize> = Matrix<f64, Const<D>, Dyn, VecStorage<f64, Const<D>, Dyn>>;

/// Convert type to svector with D entries
pub trait IntoSVector<const D: usize> {
    /// Returns svector of fixed size D
    fn into_svector(self) -> SVector<f64, D>;
}

impl<const D: usize> IntoSVector<D> for &Mat<f64> {
    fn into_svector(self) -> SVector<f64, D> {
        let x_slice = self.col(0).subrows(0, D).try_as_slice().unwrap();
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
}
