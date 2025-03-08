use crate::{
    Error, Grid, LinearEqualityConstraint, MatrixDRows, NLPSolver, NLPSolverConstraints,
    NLPSolverOptions, Optimalities, Optimality, OptimalityMeasures, Result,
    interior_point_method::{InequalityConstraint, NLPBound, VoronoiConstraint},
};
use faer_ext::IntoFaer;
use nalgebra::{DMatrix, DVector, SVector};
use rayon::prelude::*;
use std::{fmt::Display, iter::zip, sync::Arc};

#[derive(Debug, Clone, PartialEq)]
pub struct DesignCrit {
    collapse: f64,
    filter: f64,
}

impl Default for DesignCrit {
    fn default() -> Self {
        Self {
            collapse: 1e-4,
            filter: 1e-4,
        }
    }
}

impl DesignCrit {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Contains the column-orientated support vectors and their weights. Additionally the crit is used
/// to set the filter and collapse constraints.
#[derive(Debug, Clone, PartialEq)]
pub struct Design<const D: usize> {
    /// The i-th weights element relates to the i-th column in support vector matrix.
    pub weights: DVector<f64>,
    /// The support matrix contains the set of column-orientated support vectors.
    pub supp: MatrixDRows<D>,
    crit: DesignCrit,
}

impl<const D: usize> Display for Design<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // columns
        // weight, support vector
        let rows: String = zip(self.weights.as_slice(), self.supp.column_iter())
            .map(|(w, s)| {
                let s_values = s
                    .iter()
                    .map(|v| format!("{:>+.4}", v))
                    .collect::<Vec<String>>()
                    .join(", ");
                let s_vec = format!("[ {} ]", s_values);
                format!("{:.4}\t{}", w, s_vec)
            })
            .collect::<Vec<String>>()
            .join("\n");
        let header = "Weight\tSupport Vector".to_string();
        write!(
            f,
            "{:-^1$}\n{header}\n{rows}",
            " Design ".to_string(),
            (12 + D * 7 + (D - 1) * 2).max(28)
        )
    }
}

impl<const D: usize> Design<D> {
    /// Creates design by providing the support matrix and its weights.
    pub fn new(weights: DVector<f64>, supp: MatrixDRows<D>) -> Result<Self> {
        let weights_shape = weights.shape();
        let supp_shape = supp.shape();
        if weights_shape.0 != supp_shape.1 {
            return Err(Error::ShapeMismatch {
                mat1: "weights",
                mat2: "supp",
                dim1: 0,
                dim2: 1,
                shape1: weights_shape,
                shape2: supp_shape,
            });
        }
        let crit = DesignCrit::new();
        Ok(Self {
            weights,
            supp,
            crit,
        })
    }

    /// Creates design with equally distributed weights by providing the support matrix.
    pub fn new_from_supp(supp: MatrixDRows<D>) -> Self {
        let weights = DVector::from_element(supp.ncols(), 1.0 / (supp.ncols() as f64));
        let crit = DesignCrit::new();
        Self {
            weights,
            supp,
            crit,
        }
    }

    /// Returns design with given collapse criterium.
    pub fn with_collapse_crit(mut self, val: f64) -> Self {
        self.crit.collapse = val;
        self
    }

    /// Returns design with given filter criterium.
    pub fn with_filter_crit(mut self, val: f64) -> Self {
        self.crit.filter = val;
        self
    }

    /// Normalize weights.
    fn normalize_weights(&mut self) {
        if !self.weights.is_empty() {
            self.weights /= self.weights.sum();
        }
    }

    /// Adds a new support vector and its weights to the design.
    fn push(&mut self, supp: SVector<f64, D>, weight: f64) {
        let supp_size = self.supp.ncols();
        self.supp = self.supp.clone().insert_columns(supp_size, 1, 0.);
        self.supp.column_mut(supp_size).copy_from(&supp);
        self.weights = self.weights.push(weight);
    }

    /// Returns the most weighted support vector and its weight
    fn most_weighted_supp(&self) -> (SVector<f64, D>, f64) {
        let (id, w) = self.weights.argmax();
        (self.supp.column(id).into(), w)
    }

    /// Filters out support vectors with a weight lower lower equal to the filter criterium and
    /// normalizes the design afterwards.
    fn filter(&mut self) {
        let w = self.weights.clone();
        let filtered =
            zip(w.iter(), self.supp.column_iter()).filter(|(w, _)| **w > self.crit.filter);
        let weights = filtered.clone().map(|(w, _)| *w).collect::<Vec<_>>();
        let supp = &filtered.map(|(_, s)| s).collect::<Vec<_>>();
        match weights.len() {
            0 => {
                self.weights = DVector::zeros(0);
                self.supp = MatrixDRows::zeros(0);
            }
            _ => {
                self.weights = DVector::from_vec(weights);
                self.supp = MatrixDRows::from_columns(supp);
                self.normalize_weights()
            }
        }
    }

    /// Collapses all support vectors and adds their weights to the most weighted neighbors within a
    /// given collapse crit euclidian distance.
    pub fn collapse(&mut self) {
        let mut clusters: Vec<Design<D>> = vec![];
        zip(self.supp.column_iter(), self.weights.iter()).for_each(|(c, w)| {
            let mut i = 0;
            let mut append_to_cluster = false;
            while i < clusters.len() && !append_to_cluster {
                let cluster = clusters.get_mut(i).unwrap();
                let mut j = 0;
                while j < cluster.supp.ncols() && !append_to_cluster {
                    let cj = cluster.supp.column(j);
                    let distance = c - cj;
                    if distance.norm() < self.crit.collapse {
                        append_to_cluster = true;
                        cluster.push(c.into(), *w);
                    }
                    j += 1;
                }
                i += 1;
            }
            if !append_to_cluster {
                let supp = MatrixDRows::<D>::from_column_slice(c.as_slice());
                let weights = DVector::from_vec(vec![*w]);
                let cluster = Self::new(weights, supp).unwrap();
                clusters.push(cluster);
            }
        });
        let (supp, weights): (MatrixDRows<D>, DVector<f64>) = clusters
            .iter()
            .map(|c| (c.most_weighted_supp().0, c.weights.sum()))
            .unzip();
        self.supp = supp;
        self.weights = weights;
    }
}

/// Contains the lower and upper bound for design support vectors.
#[derive(Debug, PartialEq)]
pub struct DesignBound<const D: usize> {
    lower: SVector<f64, D>,
    upper: SVector<f64, D>,
}

impl<const D: usize> DesignBound<D> {
    /// Creates the design bound.
    pub fn new(lower: SVector<f64, D>, upper: SVector<f64, D>) -> Result<Self> {
        let (dim, distance) = (upper - lower).argmin();
        if distance <= 0.0 {
            return Err(Error::MinDistanceBetweenVectors {
                vector1: "lower",
                vector2: "upper",
                dim,
                distance,
                gt_distance: 0.0,
            });
        }

        Ok(Self { lower, upper })
    }
}

/// Which constraint is applied to design in optimal design.
pub enum DesignConstraint<const D: usize> {
    /// Use cubic design bound.
    Bound(DesignBound<D>),
}

/// Stop criteria of optimal design definition by stopping the algorithm
pub struct OptimalDesignCriteria {
    /// Stop when the relative support vectors norm distance between two
    /// iterations is lower than given value
    pub supp_precision: f64,
    /// Stop when the relative measure difference between two two
    /// iterations is lower than given value
    pub measure_precision: f64,
    /// Maximal number of iterations
    pub max_iter: usize,
}

impl Default for OptimalDesignCriteria {
    fn default() -> Self {
        Self {
            supp_precision: 1e-5,
            measure_precision: 1e-5,
            max_iter: 100,
        }
    }
}

/// Optimal Design Solver
///
/// This solver implements the "Adaptive grid Semidefinite Programming for finding optimal
/// designs" method. [Read more](https://doi.org/10.1007/s11222-017-9741-y)
pub struct OptimalDesign<const D: usize> {
    optimalities: Optimalities<D>,
    weights: Vec<f64>,
    design: Design<D>,
    constraint: DesignConstraint<D>,
    criteria: OptimalDesignCriteria,
    iterations: usize,
}

impl<const D: usize> Display for OptimalDesign<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let meas_title = format!(
            "{:-^1$}",
            " Statistics ".to_string(),
            (12 + D * 7 + (D - 1) * 2).max(28)
        );
        let footer = format!("{:-^1$}", String::new(), (12 + D * 7 + (D - 1) * 2).max(28));
        let opt_measure = format!("Optimality measure: {:.6}", self.measure());
        let design_len = format!("No. support vectors: {}", self.design().weights.len());
        let iterations = format!("Iterations: {}", self.iterations);
        write!(
            f,
            "{}\n{}\n{}\n{}\n{}\n{}",
            self.design(),
            meas_title,
            opt_measure,
            design_len,
            iterations,
            footer
        )
    }
}

impl<const D: usize> Default for OptimalDesign<D> {
    fn default() -> Self {
        let optimalities = vec![];
        let weights = vec![1.];
        let lower = SVector::<f64, D>::zeros();
        let mut upper = SVector::<f64, D>::zeros();
        upper.fill(1.);
        let mut q = SVector::<usize, D>::zeros();
        q.fill(2);
        let grid = Grid::new(lower, upper, q).unwrap();
        let design = Design::new_from_supp(grid.points);
        let bound = DesignBound::new(lower, upper).unwrap();
        let constraint = DesignConstraint::Bound(bound);
        let criteria = OptimalDesignCriteria::default();
        let iterations = 0;
        Self {
            optimalities,
            weights,
            design,
            constraint,
            criteria,
            iterations,
        }
    }
}

impl<const D: usize> OptimalDesign<D> {
    /// Returns the solver by proving a desired optimality
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the initialized solver with given optimalities
    pub fn with_optimalities(mut self, optimalities: Optimalities<D>, weights: Vec<f64>) -> Self {
        self.optimalities = optimalities;
        self.weights = weights;
        self
    }

    /// Returns the initialized solver with given optimality
    pub fn with_optimality(mut self, optimality: Arc<dyn Optimality<D> + Send + Sync>) -> Self {
        let optimalities = vec![optimality];
        self.optimalities = optimalities;
        self.weights = vec![1.];
        self
    }

    /// Returns the initialized solver with given design bound arguments
    /// of lower and upper bound
    pub fn with_bound_args(
        mut self,
        lower: SVector<f64, D>,
        upper: SVector<f64, D>,
    ) -> Result<Self> {
        let bound = DesignBound::new(lower, upper)?;
        let constraint = DesignConstraint::Bound(bound);
        self.constraint = constraint;
        Ok(self)
    }

    /// Returns the initialized solver with given design bound
    pub fn with_bound(mut self, bound: DesignBound<D>) -> Self {
        self.constraint = DesignConstraint::Bound(bound);
        self
    }

    /// Returns the initialized solver with given initial design
    /// created from grid
    pub fn with_init_design_grid(mut self, grid: Grid<D>) -> Self {
        self.design = Design::new_from_supp(grid.points);
        self
    }

    /// Returns the initialized solver with given init design
    pub fn with_init_design_grid_args(
        mut self,
        lower: SVector<f64, D>,
        upper: SVector<f64, D>,
        q: SVector<usize, D>,
    ) -> Result<Self> {
        let grid = Grid::new(lower, upper, q)?;
        let design = Design::new_from_supp(grid.points);
        self.design = design;
        Ok(self)
    }

    /// Returns the initialized solver with given init design
    pub fn with_init_design(mut self, design: Design<D>) -> Self {
        self.design = design;
        self
    }

    /// Returns the initialized solver with given [OptimalDesignCriteria]
    pub fn with_criteria(mut self, criteria: OptimalDesignCriteria) -> Self {
        self.criteria = criteria;
        self
    }

    /// Returns design of recent state of OptimalDesign algorithm
    pub fn design(&self) -> &Design<D> {
        &self.design
    }

    /// Returns the optimal design of the sum of weighted optimalities by maximizing their weighted
    /// measures.
    pub fn solve(&mut self) -> &Design<D> {
        self.minimize_weights();

        let mut iter: usize = 0;
        let mut measure_proof = true;
        let mut distance_proof = true;
        while iter < self.criteria.max_iter && measure_proof && distance_proof {
            iter += 1;
            let old_design = self.design.clone();
            let old_measure = self.measure();
            let old_supp_size = old_design.weights.len();
            self.minimize_supp();
            let supp_diff = (&old_design.supp - &self.design.supp).norm();
            self.design.collapse();
            self.minimize_weights();

            let measure = self.measure();
            let supp_size = self.design.weights.len();

            if measure.is_nan()
                || supp_size >= old_supp_size
                    && (measure <= old_measure
                        || (measure - old_measure).abs()
                            < self.criteria.measure_precision * old_measure)
            {
                measure_proof = false;
                self.design = old_design;
            } else if supp_diff < self.design.supp.norm() * self.criteria.supp_precision {
                distance_proof = false;
            }
        }
        self.iterations = iter;
        &self.design
    }

    fn minimize_weights(&mut self) {
        self._minimize_weights();
        self.design.filter();
        self._minimize_weights();
    }

    #[cfg_attr(doc, katexit::katexit)]
    /// Returns the optimality measure with respect to the design state of the solver
    ///
    /// The measure is calculated by the weighted sum of optimality measures: $\sum_i w_i
    /// \cdot \mathcal{Opt}_i$
    pub fn measure(&self) -> f64 {
        let supp = Arc::new(self.design.supp.clone());
        let weights = self
            .design
            .weights
            .view_range(.., ..)
            .into_faer()
            .to_owned();
        zip(&self.optimalities, &self.weights)
            .map(|(opt, &weight)| {
                let m = opt.measure(&weights, supp.clone());
                m * weight
            })
            .sum()
    }

    fn _minimize_weights(&mut self) {
        let size = self.design.weights.len();
        let bound = Some(NLPBound::new(
            DVector::from_element(size, 0.),
            DVector::from_element(size, 1.),
        ));
        let lin_equal = Some(LinearEqualityConstraint {
            mat: DMatrix::from_element(1, size, 1.),
        });
        let constraints = NLPSolverConstraints {
            bound,
            lin_equal,
            inequal: None,
        };
        let options = NLPSolverOptions::new();

        let supp = Arc::new(self.design.supp.clone());

        let mut design_measures = OptimalityMeasures::new();
        zip(&self.optimalities, &self.weights).for_each(|(opt, &w)| {
            let m = opt.matrix_mean(supp.clone());
            design_measures.push(m, w);
        });

        let solver = NLPSolver::new(options, constraints, Arc::new(design_measures));
        let x0 = DVector::from_element(size, 1. / size as f64);
        self.design.weights = solver.minimize(x0);
    }

    fn minimize_supp(&mut self) {
        let supp_size = self.design.supp.ncols();
        let supp_vec = (0..supp_size)
            .into_par_iter()
            .map(|x_id| self.minimize_supp_x(self.design.supp.column(x_id).into(), x_id))
            .collect::<Vec<DVector<f64>>>();
        let supp_slice: Vec<f64> = supp_vec
            .iter()
            .flat_map(|slice| slice.iter().cloned())
            .collect();
        let supp: MatrixDRows<D> = MatrixDRows::from_vec(supp_slice);
        self.design.supp = supp;
    }

    fn minimize_supp_x(&self, x0: SVector<f64, D>, x_id: usize) -> DVector<f64> {
        let bound = match &self.constraint {
            DesignConstraint::Bound(b) => Some(NLPBound::new(
                DVector::from_column_slice(b.lower.as_slice()),
                DVector::from_column_slice(b.upper.as_slice()),
            )),
        };

        let inequal = Some(InequalityConstraint::Voronoi(VoronoiConstraint::new(
            self.design.supp.view_range(.., ..).into_faer().to_owned(),
            x_id,
        )));

        let constraints = NLPSolverConstraints {
            bound,
            lin_equal: None,
            inequal,
        };
        let options = NLPSolverOptions::new();

        let supp = Arc::new(self.design.supp.clone());
        let weights = self
            .design
            .weights
            .view_range(.., ..)
            .into_faer()
            .to_owned();

        let mut design_measures = OptimalityMeasures::new();
        zip(&self.optimalities, &self.weights).for_each(|(opt, &w)| {
            let m = opt.dispersion_function(supp.clone(), weights.clone(), x_id);
            design_measures.push(m, w);
        });

        let solver = NLPSolver::new(options, constraints, Arc::new(design_measures));
        let x0 = DVector::from_column_slice(x0.as_slice());
        solver.minimize(x0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MatrixDRows, Result};
    use nalgebra::{DVector, Vector2};

    const EQ_EPS: f64 = 1e-5;
    const EQ_MAX_REL: f64 = 1e-5;

    #[test]
    fn test_design_collapse() -> Result<()> {
        let supp =
            MatrixDRows::<2>::from_vec(vec![1., 1., 2., 2., 1.00001, 1.0, 1.000001, 1.000001]);
        let weights = DVector::from(vec![0.3, 0.2, 0.4, 0.1]);
        let mut design = Design::new(weights, supp)?.with_collapse_crit(1e-4);
        design.collapse();

        let supp_rslt = MatrixDRows::<2>::from_vec(vec![1.00001, 1.0, 2.0, 2.0]);
        let weights_rslt = DVector::from(vec![0.8, 0.2]);
        assert_eq!(design.supp, supp_rslt);
        assert!(weights_rslt.relative_eq(&design.weights, EQ_EPS, EQ_MAX_REL));
        Ok(())
    }

    #[test]
    fn test_design_most_weighted() -> Result<()> {
        let supp = MatrixDRows::<2>::from_vec(vec![1., 1., 2., 2.]);
        let weights = DVector::from(vec![0.3, 0.2]);
        let design = Design::new(weights, supp)?;
        let most_weighted_supp = design.most_weighted_supp();

        assert_eq!(most_weighted_supp, (SVector::<f64, 2>::new(1., 1.), 0.3));
        Ok(())
    }

    #[test]
    fn test_design_filter() -> Result<()> {
        let supp =
            MatrixDRows::<2>::from_vec(vec![1., 1., 2., 2., 1.00001, 1.0, 1.000001, 1.000001]);
        let weights = DVector::from(vec![0.3, 0.2, 0.4, 0.1]);
        let mut design = Design::new(weights, supp)?.with_filter_crit(0.5);
        design.filter();
        assert_eq!(design.weights, DVector::zeros(0));
        let supp = MatrixDRows::<2>::zeros(0);
        assert_eq!(design.supp, supp);
        Ok(())
    }

    #[test]
    fn test_design_shape_mismatch() -> Result<()> {
        let supp = MatrixDRows::<2>::zeros(1);
        let weights = DVector::zeros(0);
        let design = Design::new(weights, supp);
        let design_err = Err(Error::ShapeMismatch {
            mat1: "weights",
            mat2: "supp",
            dim1: 0,
            dim2: 1,
            shape1: (0, 1),
            shape2: (2, 1),
        });
        assert_eq!(design, design_err);
        Ok(())
    }

    #[test]
    fn test_design_bound_shape_mismatch() -> Result<()> {
        let lower = Vector2::new(-1., -1.);
        let upper = Vector2::new(1., -2.);
        let design = DesignBound::new(lower, upper);
        let design_err = Err(Error::MinDistanceBetweenVectors {
            vector1: "lower",
            vector2: "upper",
            dim: 1,
            distance: -1.,
            gt_distance: 0.0,
        });
        assert_eq!(design, design_err);
        Ok(())
    }

    #[test]
    fn test_design_empty() -> Result<()> {
        let supp = MatrixDRows::<2>::zeros(0);
        let weights = DVector::zeros(0);
        let mut design = Design::new(weights, supp)?.with_filter_crit(0.5);
        design.filter();
        design.collapse();
        assert_eq!(design.weights, DVector::zeros(0));
        let supp = MatrixDRows::<2>::zeros(0);
        assert_eq!(design.supp, supp);
        Ok(())
    }

    #[test]
    fn test_optimal_design_display() -> Result<()> {
        let od = OptimalDesign::<1>::new();
        let od_display = format!("{od}");
        assert_eq!(
            od_display,
            "---------- Design ----------\n\
            Weight\tSupport Vector\n\
            0.5000\t[ +0.0000 ]\n\
            0.5000\t[ +1.0000 ]\n\
            -------- Statistics --------\n\
            Optimality measure: -0.000000\n\
            No. support vectors: 2\n\
            Iterations: 0\n\
            ----------------------------"
                .to_string()
        );
        Ok(())
    }
}
