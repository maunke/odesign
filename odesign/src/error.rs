/// Main error type
#[derive(thiserror::Error, Debug, PartialEq)]
pub enum Error {
    /// A vector contains a minimum value lower than required.
    #[error("minimal value for each value in vector {vector} is {ge_value}")]
    MinValue {
        /// Vector name
        vector: &'static str,
        /// Value constraint
        dim: usize,
        /// Value at position dim
        value: usize,
        /// Required minimum value
        ge_value: usize,
    },
    /// Minimal dimensional distance requirement is violated.
    #[error(
        "min distance between vectors {vector1} and {vector2} found on dim {dim} with {distance} but should > {gt_distance}"
    )]
    MinDistanceBetweenVectors {
        /// Name of vector1.
        vector1: &'static str,
        /// Name of vector2.
        vector2: &'static str,
        /// Dimension on which the distance requirement is violated.
        dim: usize,
        /// Found distance between dim-values of vector1 and vector2.
        distance: f64,
        /// Minimal required distance.
        gt_distance: f64,
    },
    /// Two given tensors do not have the same shape.
    #[error(
        "mat {mat1} with shape {shape1:?} and {mat2} with {shape2:?} have different len on dim {dim1} and {dim2}"
    )]
    ShapeMismatch {
        /// Name of matrix 1.
        mat1: &'static str,
        /// Name of matrix .
        mat2: &'static str,
        /// Affected shape of dimension on matrix 1.
        dim1: usize,
        /// Affected shape of dimension on matrix 2.
        dim2: usize,
        /// Shape of matrix 1.
        shape1: (usize, usize),
        /// Shape of matrix 2.
        shape2: (usize, usize),
    },
}

/// Main result type
pub type Result<T> = std::result::Result<T, Error>;
