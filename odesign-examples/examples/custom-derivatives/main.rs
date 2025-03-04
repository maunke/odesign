use nalgebra::{Matrix1, SVector, Vector1};
use odesign::{DOptimality, Feature, FeatureSet, LinearModel, OptimalDesign, Result};
use std::sync::Arc;

struct Monomial {
    i: i32,
}

// feature: x^i
impl Feature<1> for Monomial {
    fn val(&self, x: &SVector<f64, 1>) -> f64 {
        x[0].powi(self.i)
    }

    fn val_grad(&self, x: &SVector<f64, 1>) -> (f64, SVector<f64, 1>) {
        let val = self.val(x);
        let grad = Vector1::new(self.i as f64 * x[0].powi(self.i - 1));
        (val, grad)
    }

    fn val_grad_hes(
        &self,
        x: &SVector<f64, 1>,
    ) -> (f64, SVector<f64, 1>, nalgebra::SMatrix<f64, 1, 1>) {
        let (val, grad) = self.val_grad(x);
        (
            val,
            grad,
            Matrix1::new((self.i as f64) * (self.i as f64) * x[0].powi(self.i - 2)),
        )
    }
}

// f(x): 1 + x + x^2 + x^3
fn main() -> Result<()> {
    // define set of features
    let mut fs = FeatureSet::new();
    for i in 0..4 {
        let c: Arc<_> = Monomial { i }.into();
        fs.push(c);
    }

    // define linear model with features
    let lm = LinearModel::new(fs.features);

    // define optimality, bound and init design args
    let optimality = Arc::new(DOptimality::new(lm.into()));
    let lower = Vector1::new(-1.0);
    let upper = Vector1::new(1.0);
    let q = Vector1::new(101);

    // define Optimal Design resolver
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_bound_args(lower, upper)?
        .with_init_design_grid_args(lower, upper, q)?;

    // equivalent definition of optimal design solver
    //
    // let bound = odesign::DesignBound::new(lower, upper)?;
    // let init_grid = odesign::Grid::new(lower, upper, q)?;
    // let init_design = odesign::Design::new_from_supp(init_grid.points);
    // let mut od = OptimalDesign::new()
    //     .with_optimality(optimality)
    //     .with_bound(bound)
    //     .with_init_design(init_design);

    // find optimal design
    // get design by: let design = od.solve();
    // or after solving: let design = od.design();
    od.solve();

    // display optimal design
    println!("{od}");

    Ok(())
}
