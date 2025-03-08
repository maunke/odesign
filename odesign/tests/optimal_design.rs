use nalgebra::{DVector, SVector, Vector1};
use num_dual::*;
use odesign::{
    AOptimality, COptimality, DOptimality, Design, DesignBound, Feature, FeatureFunction,
    FeatureSet, Grid, LinearModel, MatrixDRows, OptimalDesign, OptimalDesignCriteria, Optimality,
    Result,
};
use std::sync::Arc;

const EQ_EPS: f64 = 1e-5;
const EQ_MAX_REL: f64 = 1e-5;

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

#[test]
fn test_optimal_design_cdcrit_poly_2() -> Result<()> {
    let mut fs = FeatureSet::new();
    for i in 0..3 {
        let c: Arc<_> = Monomial { i }.into();
        fs.push(c);
    }

    let lm: Arc<_> = LinearModel::new(fs.features).into();

    let q: SVector<usize, 1> = Vector1::new(101);
    let lower = Vector1::new(-1.);
    let upper = Vector1::new(1.);
    let d_opt = Arc::new(DOptimality::new(lm.clone()));
    let c = DVector::from_vec(vec![0., 0., 1.]);
    let c_opt = COptimality::new(lm.clone(), c.into())?;
    let c_opt = Arc::new(c_opt);
    let optimalities: Vec<Arc<dyn Optimality<1> + Send + Sync>> = vec![d_opt, c_opt];
    let weights = vec![1., 1.];
    let init_grid = Grid::new(lower, upper, q)?;
    let init_design = Design::new_from_supp(init_grid.points);
    let criteria = OptimalDesignCriteria::default();
    let mut od = OptimalDesign::new()
        .with_optimalities(optimalities, weights)
        .with_bound_args(lower, upper)?
        .with_init_design(init_design)
        .with_criteria(criteria);
    let design = od.solve();

    let weights_rslt = DVector::from_vec(vec![0.3, 0.4, 0.3]);
    let supp_rslt = MatrixDRows::from_vec(vec![-1., 0., 1.]);

    assert!(
        design
            .weights
            .relative_eq(&weights_rslt, EQ_EPS, EQ_MAX_REL)
    );
    assert!(design.supp.relative_eq(&supp_rslt, EQ_EPS, EQ_MAX_REL));
    Ok(())
}

#[test]
fn test_optimal_design_dcrit_poly_3() -> Result<()> {
    let mut fs = FeatureSet::new();

    for i in 0..4 {
        let c: Arc<_> = Monomial { i }.into();
        fs.push(c);
    }

    let lm = LinearModel::new(fs.features);

    let q: SVector<usize, 1> = Vector1::new(101);
    let lower = Vector1::new(-1.);
    let upper = Vector1::new(1.);
    let optimality = Arc::new(DOptimality::new(lm.into()));
    let bound = DesignBound::new(lower, upper)?;
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_bound(bound)
        .with_init_design_grid_args(lower, upper, q)?;
    let design = od.solve();

    let weights_rslt = DVector::from_element(4, 0.25);
    let supp_rslt = MatrixDRows::from_vec(vec![-1., -0.45, 0.45, 1.]);

    assert!(
        design
            .weights
            .relative_eq(&weights_rslt, EQ_EPS, EQ_MAX_REL)
    );
    assert!(design.supp.relative_eq(&supp_rslt, EQ_EPS, EQ_MAX_REL));
    Ok(())
}

#[test]
fn test_optimal_design_dcrit_poly_1() -> Result<()> {
    let mut fs = FeatureSet::new();
    for i in 0..2 {
        let c: Arc<_> = Monomial { i }.into();
        fs.push(c);
    }

    let lm = LinearModel::new(fs.features);

    let q: SVector<usize, 1> = Vector1::new(101);
    let lower = Vector1::new(-1.);
    let upper = Vector1::new(1.);
    let optimality = Arc::new(DOptimality::new(lm.into()));
    let grid = Grid::new(lower, upper, q)?;
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_bound_args(lower, upper)?
        .with_init_design_grid(grid);
    let design = od.solve();

    let weights_rslt = DVector::from_element(2, 0.5);
    let supp_rslt = MatrixDRows::from_vec(vec![-1., 1.]);

    assert!(
        design
            .weights
            .relative_eq(&weights_rslt, EQ_EPS, EQ_MAX_REL)
    );
    assert!(design.supp.relative_eq(&supp_rslt, EQ_EPS, EQ_MAX_REL));
    Ok(())
}

#[test]
fn test_optimal_design_ccrit() -> Result<()> {
    let mut fs = FeatureSet::new();
    for i in 0..3 {
        let c: Arc<_> = Monomial { i }.into();
        fs.push(c);
    }

    let lm = LinearModel::new(fs.features);

    let q: SVector<usize, 1> = Vector1::new(101);
    let lower = Vector1::new(-1.);
    let upper = Vector1::new(1.);
    let c = DVector::from_vec(vec![0., 0., 1.]);
    let c_opt = COptimality::new(lm.into(), c.into())?;
    let optimality = Arc::new(c_opt);
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_bound_args(lower, upper)?
        .with_init_design_grid_args(lower, upper, q)?;
    let design = od.solve();

    let weights_rslt = DVector::from_vec(vec![0.25, 0.50, 0.25]);
    let supp_rslt = MatrixDRows::from_vec(vec![-1., 0., 1.]);

    assert!(
        design
            .weights
            .relative_eq(&weights_rslt, EQ_EPS, EQ_MAX_REL)
    );
    assert!(design.supp.relative_eq(&supp_rslt, EQ_EPS, EQ_MAX_REL));
    Ok(())
}

#[test]
fn test_optimal_design_acrit_poly_1() -> Result<()> {
    let mut fs = FeatureSet::new();
    for i in 0..3 {
        let c: Arc<_> = Monomial { i }.into();
        fs.push(c);
    }

    let lm = LinearModel::new(fs.features);

    let q: SVector<usize, 1> = Vector1::new(101);
    let lower = Vector1::new(-1.);
    let upper = Vector1::new(1.);
    let optimality = Arc::new(AOptimality::new(lm.into()));
    let grid = Grid::new(lower, upper, q)?;
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_bound_args(lower, upper)?
        .with_init_design_grid(grid);
    let design = od.solve();

    let weights_rslt = DVector::from_vec(vec![0.25, 0.5, 0.25]);
    let supp_rslt = MatrixDRows::from_vec(vec![-1., 0., 1.]);

    assert!(
        design
            .weights
            .relative_eq(&weights_rslt, EQ_EPS, EQ_MAX_REL)
    );
    assert!(design.supp.relative_eq(&supp_rslt, EQ_EPS, EQ_MAX_REL));
    Ok(())
}
