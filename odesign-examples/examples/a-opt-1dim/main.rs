use nalgebra::{SVector, Vector1};
use num_dual::DualNum;
use odesign::{
    AOptimality, Feature, FeatureFunction, FeatureSet, LinearModel, OptimalDesign, Result,
};

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
struct Exp {}

impl FeatureFunction<1> for Exp {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 1>) -> D {
        (x[0]).exp().powi(-1)
    }
}

// f(x): 1 + x + x^-1 + exp(-x)
fn main() -> Result<()> {
    // define set of features
    let mut fs = FeatureSet::new();
    for i in -1..2 {
        let c = Monomial { i };
        fs.push(c);
    }
    let c = Exp {};
    fs.push(c);

    // define linear model with features
    let lm = LinearModel::from(fs);

    // define optimality, bound and init design args
    let optimality = AOptimality::new(lm.into());
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
    // or: let design = od.design();
    od.solve();

    // display optimal design
    println!("{od}");

    Ok(())
}
