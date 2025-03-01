use crate::{Error, Result};
use nalgebra::{Const, Dyn, Matrix, SVector, VecStorage};

/// Cubic grid defined by lower/upper bound and the dimensional sample number q.
#[derive(Debug, PartialEq)]
pub struct Grid<const D: usize> {
    /// Lower bound.
    pub lower: SVector<f64, D>,
    /// Upper bound.
    pub upper: SVector<f64, D>,
    /// Sample number per dimension.
    pub q: SVector<usize, D>,
    /// Set of points in grid.
    pub points: Matrix<f64, Const<D>, Dyn, VecStorage<f64, Const<D>, Dyn>>,
}

impl<const D: usize> Grid<D> {
    /// Creates cubic grid with help of generating the cartesian product.
    pub fn new(
        lower: SVector<f64, D>,
        upper: SVector<f64, D>,
        q: SVector<usize, D>,
    ) -> Result<Self> {
        let (q_min_dim, q_min) = q.argmin();
        if q_min < 2 {
            return Err(Error::MinValue {
                vector: "q",
                dim: q_min_dim,
                value: q_min,
                ge_value: 2,
            });
        }
        let (dim, distance) = (upper - lower).argmin();
        if distance <= 0.0 {
            return Err(Error::MinDistanceBetweenVectors {
                vector1: "lower",
                vector2: "upper",
                dim,
                distance,
                gt_distance: 0.0,
            });
        }

        let q_f = SVector::<f64, D>::from_vec(
            q.iter()
                .map(|v| 1. / (*v as f64 - 1.))
                .collect::<Vec<f64>>(),
        );
        let delta = (upper - lower).component_mul(&q_f);
        let pos = SVector::<usize, D>::zeros();
        let x = SVector::<f64, D>::zeros();
        let points_vec = Grid::build_grid(&lower, &delta, &q, pos, x, 0);
        let points =
            Matrix::<f64, Const<D>, Dyn, VecStorage<f64, Const<D>, Dyn>>::from_columns(&points_vec);
        Ok(Self {
            lower,
            upper,
            q,
            points,
        })
    }

    fn build_grid(
        lower: &SVector<f64, D>,
        delta: &SVector<f64, D>,
        q: &SVector<usize, D>,
        mut pos: SVector<usize, D>,
        mut x: SVector<f64, D>,
        d: usize,
    ) -> Vec<SVector<f64, D>> {
        if d < lower.len() {
            let mut vec: Vec<SVector<f64, D>> = vec![];
            for i in 0..q[d] {
                x[d] = lower[d] + i as f64 * delta[d];
                pos[d] = i;
                vec.extend(Grid::build_grid(lower, delta, q, pos, x, d + 1));
            }
            vec
        } else {
            vec![x]
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Error, Grid, MatrixDRows, Result};
    use nalgebra::Vector2;

    #[test]
    fn grid() -> Result<()> {
        let lower = Vector2::new(0., 0.);
        let upper = Vector2::new(1., 1.);
        let q = Vector2::new(2, 2);
        let grid = Grid::new(lower, upper, q)?;

        let grid_rslt: MatrixDRows<2> = MatrixDRows::from_vec(vec![0., 0., 0., 1., 1., 0., 1., 1.]);
        assert_eq!(grid.points, grid_rslt);
        Ok(())
    }

    #[test]
    fn grid_bound() -> Result<()> {
        let lower = Vector2::new(0., 0.);
        let upper = Vector2::new(1., 0.);
        let q = Vector2::new(2, 2);
        let grid = Grid::new(lower, upper, q);

        let grid_err = Err(Error::MinDistanceBetweenVectors {
            vector1: "lower",
            vector2: "upper",
            dim: 1,
            distance: 0.,
            gt_distance: 0.,
        });
        assert_eq!(grid, grid_err);
        Ok(())
    }

    #[test]
    fn grid_sample_size() -> Result<()> {
        let lower = Vector2::new(0., 0.);
        let upper = Vector2::new(1., 1.);
        let q = Vector2::new(2, 1);
        let grid = Grid::new(lower, upper, q);

        let grid_err = Err(Error::MinValue {
            vector: "q",
            dim: 1,
            value: 1,
            ge_value: 2,
        });
        assert_eq!(grid, grid_err);
        Ok(())
    }
}
