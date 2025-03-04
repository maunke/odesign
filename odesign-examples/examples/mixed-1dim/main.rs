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

#[derive(Feature)]
#[dimension = 1]
struct Sin {
    period: f64,
}

impl FeatureFunction<1> for Sin {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 1>) -> D {
        x[0].clone().mul(self.period).sin()
    }
}

// f(x): 1 + x + 1 / x + sin(x) + sin(2x) + sin(3x) + sin(4x)
fn main() -> Result<()> {
    // define set of features
    let mut fs = FeatureSet::new();
    for i in -1..2 {
        let c: Arc<_> = Monomial { i }.into();
        fs.push(c);
    }
    for period in 1..5 {
        let c: Arc<_> = Sin {
            period: period as f64,
        }
        .into();
        fs.push(c);
    }

    // define linear model with features
    let lm = LinearModel::new(fs.features);

    // define optimality, bound and init design args
    let optimality = Arc::new(DOptimality::new(lm.into()));
    let lower = Vector1::new(0.5);
    let upper = Vector1::new(2.5);
    let q = Vector1::new(101);

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
