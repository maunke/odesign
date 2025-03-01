use nalgebra::{SVector, Vector1};
use num_dual::DualNum;
use odesign::{
    DOptimality, Feature, FeatureFunction, FeatureSet, LinearModel, OptimalDesign, Result,
};
use std::sync::Arc;

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

// f(x): 1 + x + x ^ 2
fn main() -> Result<()> {
    // define set of features
    let mut fs = FeatureSet::new();
    let c: Arc<_> = Monomial { i: 0 }.into();
    fs.push(c);
    let c: Arc<_> = Monomial { i: 1 }.into();
    fs.push(c);
    let c: Arc<_> = Monomial { i: 2 }.into();
    fs.push(c);

    // define linear model with features
    let lm = LinearModel::new(fs.features);

    // define optimality, bound and init design args
    let optimality = Arc::new(DOptimality::new(lm.into()));
    let lower = Vector1::new(-1.);
    let upper = Vector1::new(1.);
    let q: SVector<usize, 1> = Vector1::new(101);

    // define Optimal Design resolver
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_bound_args(lower, upper)?
        .with_init_design_grid_args(lower, upper, q)?;

    // find optimal design
    // get design by: let design = od.solve();
    // or after solving: let design = od.design();
    od.solve();

    // display optimal design
    // - weights
    // - support vectors
    // - optimality measure for design
    println!("{od}");

    Ok(())
}
