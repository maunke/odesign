#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

extern crate self as odesign;

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
    AOptimality, COptimality, CostsOptimality, DOptimality, Optimalities, Optimality,
    OptimalityMeasures, WeightedOptimality,
};
pub use utils::{IntoSVector, MatrixDRows, MatrixUnion};

pub use nalgebra::{
    DVector, Matrix1, Matrix2, Matrix3, Matrix4, Matrix5, Matrix6, SMatrix, SVector, Vector1,
    Vector2, Vector3, Vector4, Vector5, Vector6,
};
pub use num_dual::DualNum;

#[doc(hidden)]
pub mod __private {
    pub use ::nalgebra;
    pub use ::num_dual;
}
