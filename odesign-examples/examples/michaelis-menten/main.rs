//! Michaelis-Menten Enzyme Kinetics - D-Optimal Experimental Design
//!
//! Model: v = V_max * [S] / (K_m + [S])
//!
//! Sensitivity functions for locally optimal design:
//!   ∂v/∂V_max = [S] / (K_m + [S])
//!   ∂v/∂K_m   = - V_max * [S] / (K_m + [S])²
//!
//! For bounded designs with upper bound S₂, optimal lower point:
//!   S₁_opt = S₂ x K_m / (S₂ + 2xK_m)
//!
//! References:
//! - Duggleby (1979), J. Theoretical Biology 81:671-684
//! - Atkinson & Donev (1992), Optimum Experimental Designs, Oxford

use nalgebra::{SVector, Vector1};
use num_dual::DualNum;
use odesign::{
    DOptimality, Feature, FeatureFunction, FeatureSet, LinearModel, OptimalDesign, Result,
};
use std::sync::Arc;

const V_MAX: f64 = 1.0;
const K_M: f64 = 10.0;

/// Sensitivity to V_max: ∂v/∂V_max = [S] / (K_m + [S])
#[derive(Feature)]
#[dimension = 1]
struct SensitivityVmax;

impl FeatureFunction<1> for SensitivityVmax {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 1>) -> D {
        let s = x[0].clone();
        s.clone() / (D::from(K_M) + s)
    }
}

/// Sensitivity to K_m: ∂v/∂K_m = -V_max * [S] / (K_m + [S])²
#[derive(Feature)]
#[dimension = 1]
struct SensitivityKm;

impl FeatureFunction<1> for SensitivityKm {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 1>) -> D {
        let s = x[0].clone();
        let denom = D::from(K_M) + s.clone();
        -D::from(V_MAX) * s / (denom.clone() * denom)
    }
}

fn main() -> Result<()> {
    println!("Michaelis-Menten D-Optimal Design");
    println!("Model: v = V_max * [S] / (K_m + [S])");
    println!("V_max = {V_MAX}, K_m = {K_M}\n");

    // Theoretical optimal for bounded design: S₁ = S₂xK_m/(S₂+2K_m)
    let upper = 50.0;
    let s1_theory = upper * K_M / (upper + 2.0 * K_M);
    println!("Theoretical S₁ for S₂={upper}: {s1_theory:.4}\n");

    // Linearized model uses sensitivities as regressors
    let mut fs = FeatureSet::new();
    fs.push(Arc::new(SensitivityVmax));
    fs.push(Arc::new(SensitivityKm));

    let lm: Arc<_> = LinearModel::new(fs.features).into();
    let opt: Arc<_> = DOptimality::new(lm).into();

    let lower = Vector1::new(0.01);
    let upper = Vector1::new(50.0);

    let mut od = OptimalDesign::new()
        .with_optimality(opt)
        .with_bound_args(lower, upper)?
        .with_init_design_grid_args(lower, upper, Vector1::new(501))?;

    od.solve();
    println!("{od}");

    Ok(())
}
