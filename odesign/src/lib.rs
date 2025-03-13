#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
mod error;
mod feature;
mod grid;
mod interior_point_method;
mod linear_model;
mod optimal_design;
mod optimality;
mod utils;

pub use error::{Error, Result};
pub use feature::{Feature, FeatureFunction, FeatureSet};
pub use grid::Grid;
pub use interior_point_method::{
    LinearEqualityConstraint, NLPFunctionTarget, NLPSolver, NLPSolverConstraints, NLPSolverOptions,
};
pub use linear_model::LinearModel;
pub use odesign_derive::Feature;
pub use optimal_design::{
    CustomDesignBound, Design, DesignBound, DesignConstraint, OptimalDesign, OptimalDesignCriteria,
};
pub use optimality::{
    AOptimality, COptimality, DOptimality, Optimalities, Optimality, OptimalityMeasures,
};
pub use utils::{IntoSVector, MatrixDRows};
