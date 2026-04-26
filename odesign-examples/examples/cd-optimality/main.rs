use nalgebra::{DVector, SVector, Vector1};
use num_dual::DualNum;
use odesign::{
    COptimality, DOptimality, Feature, FeatureFunction, FeatureSet, LinearModel, OptimalDesign,
    Optimalities, Result, WeightedOptimality,
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

// f(x): 1 + x + x^2
fn main() -> Result<()> {
    // define set of features
    let mut fs = FeatureSet::new();
    for i in 0..3 {
        let c = Monomial { i };
        fs.push(c);
    }

    // define linear model with features
    let lm: Arc<_> = LinearModel::from(fs).into();
    // define optimality, bound and init design args
    let d_optimality = DOptimality::new(lm.clone());
    let c = DVector::from_vec(vec![0., 0., 1.]);
    let c_optimality = COptimality::new(lm, c.into())?;
    let optimalities: Optimalities<1> = vec![
        WeightedOptimality::new(d_optimality, 1.),
        WeightedOptimality::new(c_optimality, 1.),
    ];
    let lower = Vector1::new(-1.0);
    let upper = Vector1::new(1.0);
    let q = Vector1::new(101);

    // define Optimal Design resolver
    let mut od = OptimalDesign::new()
        .with_optimalities(optimalities)
        .with_bound_args(lower, upper)?
        .with_init_design_grid_args(lower, upper, q)?;

    od.solve();

    // display optimal design
    println!("{od}");

    Ok(())
}
