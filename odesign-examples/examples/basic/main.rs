use nalgebra::{SVector, Vector1};
use num_dual::DualNum;
use odesign::{
    DOptimality, Feature, FeatureFunction, FeatureSet, LinearModel, OptimalDesign, Result,
};

// with help of the Feature derive and the required implementation
// of the FeatureFunction the derivatives are automatically
// generated
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

// f(x): 1 + x
fn main() -> Result<()> {
    // define set of features
    let mut fs = FeatureSet::new();
    let c = Monomial { i: 0 };
    fs.push(c);
    let c = Monomial { i: 1 };
    fs.push(c);

    // define linear model with features
    let lm = LinearModel::from(fs);

    // define optimality, bound and init design args
    let optimality = DOptimality::new(lm.into());
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
    // or: let design = od.design();
    od.solve();

    // display optimal design
    println!("{od}");
    // Output
    // ---------- Design ----------
    // Weight  Support Vector
    // 0.5000  [ -1.0000 ]
    // 0.5000  [ +1.0000 ]
    // -------- Statistics --------
    // Optimality measure: 1.000000
    // No. support vectors: 2
    // Iterations: 1
    // ----------------------------

    Ok(())
}
