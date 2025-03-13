use nalgebra::{SVector, Vector1};
use num_dual::DualNum;
use odesign::{
    CustomDesignBound, DOptimality, Feature, FeatureFunction, FeatureSet, Grid, LinearModel,
    OptimalDesign, Result,
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
struct CustomDesignBoundConstraint {}

// design constraint g with g(x) <= 0,
// such that -1 <= x <= 1
impl FeatureFunction<1> for CustomDesignBoundConstraint {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 1>) -> D {
        x[0].powi(2) - 1.
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
    let q = Vector1::new(101);
    let grid = Grid::new(lower, upper, q)?;

    let design_bound_const: Arc<_> = CustomDesignBoundConstraint {}.into();
    let custom_bound = CustomDesignBound::new(design_bound_const);

    // define Optimal Design resolver
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_custom_bound(custom_bound)
        .with_init_design_grid(grid);

    od.solve();

    println!("{od}");

    Ok(())
}
