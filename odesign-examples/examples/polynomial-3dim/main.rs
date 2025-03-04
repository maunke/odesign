use nalgebra::{SVector, Vector3};
use num_dual::DualNum;
use odesign::{
    DOptimality, Feature, FeatureFunction, FeatureSet, LinearModel, OptimalDesign, Result,
};
use std::sync::Arc;

#[derive(Feature)]
#[dimension = 3]
struct Monomial {
    i: i32,
    j: i32,
    k: i32,
}

impl FeatureFunction<3> for Monomial {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 3>) -> D {
        x[0].powi(self.i) * x[1].powi(self.j) * x[2].powi(self.k)
    }
}

// f(x, y, z):  1 + x + y + z
//              + x * y + x * z + y * y
//              + x ^ 2 + y ^ 2 + z ^ 2
fn main() -> Result<()> {
    let mut fs = FeatureSet::new();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                if i + j + k < 3 {
                    let c: Arc<_> = Monomial { i, j, k }.into();
                    fs.push(c);
                }
            }
        }
    }

    let lm = LinearModel::new(fs.features);

    let q = Vector3::new(11, 11, 11);
    let lower = Vector3::new(-1., -1., -1.);
    let upper = Vector3::new(1., 1., 1.);
    let optimality = Arc::new(DOptimality::new(lm.into()));
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_bound_args(lower, upper)?
        .with_init_design_grid_args(lower, upper, q)?;
    od.solve();

    println!("{od}");

    Ok(())
}
