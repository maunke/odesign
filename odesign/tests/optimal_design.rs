use nalgebra::{DVector, SVector, Vector1};
use num_dual::*;
use odesign::{
    AOptimality, COptimality, CostsOptimality, CustomDesignBound, DOptimality, Design, DesignBound,
    Feature, FeatureFunction, FeatureSet, Grid, LinearModel, MatrixDRows, MatrixUnion,
    OptimalDesign, OptimalDesignCriteria, Optimalities, Result, WeightedOptimality,
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

#[derive(Feature)]
#[dimension = 1]
struct CustomBoundConstraint {}

impl FeatureFunction<1> for CustomBoundConstraint {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 1>) -> D {
        x[0].powi(2) - 1.0
    }
}

#[test]
fn test_optimal_design_cdcrit_poly_2() -> Result<()> {
    let mut fs = FeatureSet::new();
    for i in 0..3 {
        let c = Monomial { i };
        fs.push(c);
    }

    let lm = LinearModel::from(fs);

    let q: SVector<usize, 1> = Vector1::new(101);
    let lower = Vector1::new(-1.);
    let upper = Vector1::new(1.);
    let d_opt = DOptimality::new(lm.clone());
    let c = DVector::from_vec(vec![0., 0., 1.]);
    let c_opt = COptimality::new(lm.clone(), c)?;
    let optimalities = vec![
        WeightedOptimality::new(d_opt, 1.),
        WeightedOptimality::new(c_opt, 1.),
    ];
    let init_grid = Grid::new(lower, upper, q)?;
    let init_design = Design::new_from_supp(init_grid.points);
    let criteria = OptimalDesignCriteria::default();
    let mut od = OptimalDesign::new()
        .with_optimalities(optimalities)
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
        let c = Monomial { i };
        fs.push(c);
    }

    let lm = LinearModel::from(fs);

    let q: SVector<usize, 1> = Vector1::new(101);
    let lower = Vector1::new(-1.);
    let upper = Vector1::new(1.);
    let optimality = DOptimality::new(lm);
    let bound = DesignBound::new(lower, upper)?;
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_bound(bound)
        .with_init_design_grid_args(lower, upper, q)?;
    let design = od.solve();

    let weights_rslt = DVector::from_element(4, 0.25);
    let supp_rslt = MatrixDRows::from_vec(vec![-1., -1. / 5.0.sqrt(), 1. / 5.0.sqrt(), 1.]);

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
        let c = Monomial { i };
        fs.push(c);
    }

    let lm = LinearModel::from(fs);

    let q: SVector<usize, 1> = Vector1::new(101);
    let lower = Vector1::new(-1.);
    let upper = Vector1::new(1.);
    let optimality = DOptimality::new(lm);
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
        let c = Monomial { i };
        fs.push(c);
    }

    let lm = LinearModel::from(fs);

    let q: SVector<usize, 1> = Vector1::new(101);
    let lower = Vector1::new(-1.);
    let upper = Vector1::new(1.);
    let c = DVector::from_vec(vec![0., 0., 1.]);
    let c_opt = COptimality::new(lm, c)?;
    let mut od = OptimalDesign::new()
        .with_optimality(c_opt)
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
        let c = Monomial { i };
        fs.push(c);
    }

    let lm = LinearModel::from(fs);

    let q: SVector<usize, 1> = Vector1::new(101);
    let lower = Vector1::new(-1.);
    let upper = Vector1::new(1.);
    let optimality = AOptimality::new(lm);
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

#[test]
fn test_optimal_design_dcrit_poly_1_custom_bound() -> Result<()> {
    let mut fs = FeatureSet::new();
    for i in 0..2 {
        let c = Monomial { i };
        fs.push(c);
    }

    let lm = LinearModel::from(fs);

    let q: SVector<usize, 1> = Vector1::new(101);
    let lower = Vector1::new(-0.9);
    let upper = Vector1::new(0.9);
    let optimality = DOptimality::new(lm);
    let grid = Grid::new(lower, upper, q)?;
    let design_bound_const: Arc<_> = CustomBoundConstraint {}.into();
    let custom_bound = CustomDesignBound::new(design_bound_const);
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_custom_bound(custom_bound)
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
fn test_optimal_design_dcrit_costscrit_poly_3() -> Result<()> {
    let mut fs = FeatureSet::new();

    for i in 0..2 {
        let c = Monomial { i };
        fs.push(c);
    }

    let lm = LinearModel::from(fs);

    let q: SVector<usize, 1> = Vector1::new(101);
    let lower = Vector1::new(-1.);
    let upper = Vector1::new(1.);
    let init_grid = Grid::new(lower, upper, q)?;
    let measurements: Arc<_> = MatrixDRows::from_vec(vec![-0.91, 0.91]).into();
    let supp = init_grid.points.union(&measurements);
    let init_design = Design::new_from_supp(supp);

    let d_opt = DOptimality::new(lm);
    let alpha = 1.0;
    let costs_opt = CostsOptimality::new(measurements, alpha);
    let optimalities: Optimalities<1> = vec![
        WeightedOptimality::new(d_opt, 1.),
        WeightedOptimality::new(costs_opt, 1.),
    ];
    let bound = DesignBound::new(lower, upper)?;
    let mut od = OptimalDesign::new()
        .with_optimalities(optimalities)
        .with_bound(bound)
        .with_init_design(init_design);
    let design = od.solve();

    let weights_rslt = DVector::from_element(2, 0.5);
    let supp_rslt = MatrixDRows::from_vec(vec![-0.91, 0.91]);

    assert!(
        design
            .weights
            .relative_eq(&weights_rslt, EQ_EPS, EQ_MAX_REL)
    );
    assert!(design.supp.relative_eq(&supp_rslt, EQ_EPS, EQ_MAX_REL));
    Ok(())
}
