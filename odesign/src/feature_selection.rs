use crate::optimal_design::{DesignCrit, DiscreteDesign, Measurements, ReplicationFactor};
use crate::utils::subsets_len_k_in_s;
use crate::{
    COptimality, CostsOptimality, DOptimality, Design, DesignBound, Error, FeatureSet, LinearModel,
    MatrixUnion, OptimalDesign, Optimalities, Result, Weight, WeightRange, WeightedOptimality,
    WeightsFunction, WeightsKind,
};
use nalgebra::DVector;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::BTreeMap;
use std::sync::Arc;

const TERMINATION_STEPS: usize = 1_000;
const SFFS_LOOP_REPEAT_THRESHOLD: usize = 10;

/// Design constraint.
#[derive(Clone)]
pub enum DesignConstraint<const D: usize> {
    /// A bound constraint on the design.
    Bound(DesignBound<D>),
}

#[derive(Default)]
struct State {
    iteration: usize,
}

/// Termination criteria for the feature selection algorithm.
#[derive(Clone)]
pub struct Termination {
    cardinality: usize,
    steps: usize,
}

impl Termination {
    /// Creates a new [`Termination`] with the given `cardinality`.
    pub fn new(cardinality: usize) -> Self {
        Self {
            cardinality,
            steps: TERMINATION_STEPS,
        }
    }

    /// Sets the number of steps to terminate after.
    pub fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
}

/// Options for the cost optimality.
#[derive(Clone)]
pub struct CostOptions {
    alpha: f64,
}

impl CostOptions {
    /// Creates a new [`CostOptions`] with the given `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for CostOptions {
    fn default() -> Self {
        Self { alpha: 1. }
    }
}

/// Weights for the feature selection algorithm.
#[derive(Clone)]
pub struct FeatureSelectionWeights {
    det_weight: WeightsKind,
    cost_weight: Weight,
    max_cardinality: usize,
}

impl FeatureSelectionWeights {
    /// Creates a new [`FeatureSelectionWeights`] with the given `det_weight` and `cost_weight`.
    pub fn new(det_weight: WeightsKind, cost_weight: Weight) -> Self {
        Self {
            det_weight,
            cost_weight,
            ..Default::default()
        }
    }

    fn with_max_cardinality(mut self, max_cardinality: usize) -> Self {
        self.max_cardinality = max_cardinality;
        self
    }

    /// Computes the weight for the given `cardinality`.
    pub fn weight(&self, cardinality: usize) -> Weight {
        match &self.det_weight {
            WeightsKind::Linear(linear) => linear.weight(cardinality, self.max_cardinality),
            WeightsKind::Constant(constant) => *constant,
        }
    }
}

impl Default for FeatureSelectionWeights {
    fn default() -> Self {
        Self {
            det_weight: WeightsKind::Linear(
                WeightRange::new(
                    Weight::try_from(0.1).expect("is 0.1"),
                    Weight::try_from(0.9).expect("is 0.9"),
                )
                .expect("is valid range"),
            ),
            cost_weight: Weight::try_from(0.5).expect("is 0.5"),
            max_cardinality: 1,
        }
    }
}

type SffsBook<const D: usize> = BTreeMap<usize, LinearModel<D>>;

/// Options for the feature selection algorithm.
#[derive(Clone)]
pub struct FeatureSelectionOptions<const D: usize> {
    horizon_length: usize,
    weights: FeatureSelectionWeights,
    design_constraint: DesignConstraint<D>,
    termination: Termination,
    init_design: Design<D>,
    design_crit: DesignCrit,
    replication_factor: ReplicationFactor,
    cost: CostOptions,
}

impl<const D: usize> FeatureSelectionOptions<D> {
    /// Creates a new [`FeatureSelectionOptions`] with the given `design_constraint`,
    /// `horizon_length`, `init_design`, and `termination`.
    pub fn new(
        design_constraint: DesignConstraint<D>,
        horizon_length: usize,
        init_design: Design<D>,
        termination: Termination,
    ) -> Self {
        let weights =
            FeatureSelectionWeights::default().with_max_cardinality(termination.cardinality);
        Self {
            horizon_length,
            weights,
            design_constraint,
            termination,
            init_design,
            design_crit: DesignCrit::default(),
            replication_factor: ReplicationFactor::Fixed(1),
            cost: CostOptions::default(),
        }
    }

    /// Sets the weights for the feature selection algorithm.
    pub fn with_weights(mut self, weights: FeatureSelectionWeights) -> Self {
        self.weights = weights.with_max_cardinality(self.termination.cardinality);
        self
    }

    /// Sets the design criterion for designs within the feature selection algorithm.
    pub fn with_design_crit(mut self, design_crit: DesignCrit) -> Self {
        self.design_crit = design_crit;
        self
    }

    /// Sets the replication factor for the discrete design.
    pub fn with_replication_factor(mut self, replication_factor: ReplicationFactor) -> Self {
        self.replication_factor = replication_factor;
        self
    }

    /// Sets the cost options for the costs optimality.
    pub fn with_cost(mut self, cost: CostOptions) -> Self {
        self.cost = cost;
        self
    }
}

/// Termination reason for the feature selection algorithm.
#[derive(Clone)]
pub enum TerminationReason {
    /// The maximum number of steps has been reached.
    MaxSteps(usize),
    /// The best model has been found.
    BestModelFound,
}

/// Step in the feature selection algorithm. It marks if the algorithm has terminated or if a new
//  discrete design needs to be measured.
#[derive(Clone)]
pub enum Step<const D: usize> {
    /// The algorithm has terminated.
    Termination(TerminationReason),
    /// A new discrete design needs to be measured.
    Measure(DiscreteDesign<D>),
}

/// Optimal Design Feature Selection
pub struct FeatureSelection<const D: usize> {
    feature_set: FeatureSet<D>,
    linear_model: LinearModel<D>,
    measurements: Measurements<D>,
    state: State,
    options: FeatureSelectionOptions<D>,
    sffs_book: SffsBook<D>,
    sffs_book_trace: Vec<SffsBook<D>>,
}

impl<const D: usize> FeatureSelection<D> {
    /// Creates a new `FeatureSelection` instance.
    pub fn new(
        null_model: LinearModel<D>,
        feature_set: FeatureSet<D>,
        options: FeatureSelectionOptions<D>,
    ) -> Result<Self> {
        Self::validate_options(&null_model, &feature_set, &options)?;
        Ok(Self {
            feature_set,
            linear_model: null_model,
            measurements: Measurements::default(),
            state: State::default(),
            options,
            sffs_book: BTreeMap::default(),
            sffs_book_trace: Vec::new(),
        })
    }

    fn validate_options(
        null_model: &LinearModel<D>,
        feature_set: &FeatureSet<D>,
        options: &FeatureSelectionOptions<D>,
    ) -> Result<()> {
        // check feature set
        if feature_set.is_empty() {
            return Err(Error::EmptyArgument {
                name: "feature_set",
            });
        }
        // check dimenion dependency of null_model
        match options.design_constraint {
            DesignConstraint::Bound(design_bound) => {
                if !null_model
                    .dimension_dependency_check(design_bound.lower(), design_bound.upper())
                {
                    return Err(Error::LinearModelMissingDependency);
                }
            }
        }

        // check cardinality target
        let cardinality_target = options.termination.cardinality;
        let cardinality_null_model = null_model.n_features();
        let cardinality_feature_set = feature_set.len();
        if cardinality_target > cardinality_null_model + cardinality_feature_set {
            return Err(Error::TargetModelCardinalityUnreachable {
                cardinality_target,
                cardinality_null_model,
                cardinality_feature_set,
            });
        }
        Ok(())
    }

    /// Returns a reference to the linear model.
    pub fn linear_model(&self) -> &LinearModel<D> {
        &self.linear_model
    }

    /// Returns a reference to the measurements.
    pub fn measurements(&self) -> &Measurements<D> {
        &self.measurements
    }

    fn optimal_design(&self) -> Design<D> {
        let mut optimalities: Optimalities<D> = vec![];

        let has_measurements = self.measurements.x.ncols() > 0;

        let costs_opt_weight = match has_measurements {
            true => self.options.weights.cost_weight.get(),
            false => 0.,
        };

        let det_weight = self
            .options
            .weights
            .weight(self.linear_model.n_features())
            .get();

        let d_opt_weight = (1. - costs_opt_weight) * det_weight;

        // d-optimality with null model
        let linear_model = self.linear_model.clone();
        let d_opt = DOptimality::new(linear_model);
        optimalities.push(WeightedOptimality::new(d_opt, d_opt_weight));

        // c-optimality with horizon models
        let horizon_models = self.build_horizon_models();
        let c_opt_weight =
            (1. - costs_opt_weight) * (1. - det_weight) / (horizon_models.len() as f64);
        for hm in horizon_models {
            let features_size = hm.n_features();
            let horizon = self.options.horizon_length;
            let c = DVector::<f64>::from_fn(features_size, |idx, _| {
                if idx < features_size - horizon {
                    0.0
                } else {
                    1.0
                }
            });
            let c_opt = COptimality::new(hm, c).expect("correct coefficients set");
            optimalities.push(WeightedOptimality::new(c_opt, c_opt_weight));
        }

        // let static_supp = match has_measurements {
        //     true => self.measurements.x.clone(),
        //     false => {
        //         // calculate D-Optimal Design of Null Model
        //         let d_opt = Arc::new(DOptimality::new(self.linear_model.clone()));
        //         let bound = match &self.options.design_constraint {
        //             DesignConstraint::Bound(b) => b.clone(),
        //         };
        //         let mut od = OptimalDesign::new()
        //             .with_optimality(d_opt)
        //             .with_init_design(self.options.init_design.clone())
        //             .with_bound(bound);
        //         let design = od.solve();
        //         println!("{design}");
        //         design.supp.clone()
        //     }
        // };
        let static_supp = self.measurements.x.clone();
        if has_measurements {
            let costs_opt = CostsOptimality::new(static_supp.clone(), self.options.cost.alpha);
            optimalities.push(WeightedOptimality::new(costs_opt, costs_opt_weight));
        }

        // add to supp vectors
        let measured_supp = self.options.init_design.supp.union(&self.measurements.x);
        let mut measured_design = Design::new_from_supp(measured_supp).with_collapse_crit(1e-8);
        measured_design.collapse();
        let init_design = Design::new_from_supp(measured_design.supp.clone())
            .with_crit(self.options.design_crit.clone());

        // Extract bound from design constraint for the optimal design solver
        let bound = match &self.options.design_constraint {
            DesignConstraint::Bound(b) => *b,
        };

        let mut optimal_design = OptimalDesign::new()
            .with_optimalities(optimalities)
            .with_static_supp(static_supp)
            .with_bound(bound)
            .with_init_design(init_design);

        let result = optimal_design.solve().clone();
        let mut design = result.with_collapse_crit(1.01 * 0.04);
        design.collapse();
        design
    }

    fn build_horizon_models(&self) -> Vec<Arc<LinearModel<D>>> {
        let bound = match &self.options.design_constraint {
            DesignConstraint::Bound(b) => *b,
        };
        // build horizon models by selecting all subsets of horizon length within feature set
        let horizon_models_feature_indices =
            subsets_len_k_in_s(self.options.horizon_length, self.feature_set.len());
        let mut horizon_models = vec![];
        for set in horizon_models_feature_indices {
            let horizon_feature_set = self.feature_set.subset(set).expect("subset exists");
            let mut fs = FeatureSet::from(self.linear_model.clone());
            fs.extend(horizon_feature_set);
            let lm = LinearModel::from(fs);
            if lm.dimension_dependency_check(bound.lower(), bound.upper()) {
                horizon_models.push(lm.into());
            }
        }
        horizon_models
    }

    /// Advances the feature selection algorithm by one step. If `new_measurements` is provided,
    /// it will be added to the existing measurements.
    pub fn step(&mut self, new_measurements: Option<Measurements<D>>) -> Step<D> {
        // increment iteration
        self.state.iteration += 1;

        // check termination
        if self.state.iteration >= self.options.termination.steps {
            return Step::Termination(TerminationReason::MaxSteps(self.state.iteration));
        }

        self.forward_selection(new_measurements)
    }

    fn forward_selection(&mut self, new_measurements: Option<Measurements<D>>) -> Step<D> {
        if let Some(measurements) = new_measurements {
            // append measurements
            self.measurements.append(measurements);
            // clear sffs book since new measurements are added
            self.sffs_book.clear();
        } else {
            let design = self.optimal_design();
            let discrete_design =
                DiscreteDesign::from_design(&design, &self.options.replication_factor);
            let discrete_design_to_measure = discrete_design.diff_measurements(&self.measurements);
            if discrete_design_to_measure.replications.sum() > 0 {
                return Step::Measure(discrete_design_to_measure);
            }
        }

        let potential_models: Vec<_> = self
            .feature_set
            .iter()
            .enumerate()
            .map(|(idx, feature)| {
                let mut remaining_feature_set = self.feature_set.clone();
                remaining_feature_set.remove(idx);
                let mut lm_feature_set = FeatureSet::from(self.linear_model.clone());
                lm_feature_set.push_shared(feature.clone());
                let lm = LinearModel::from(lm_feature_set);
                (lm, remaining_feature_set)
            })
            .collect();
        let potential_models_fit = potential_models
            .into_par_iter()
            .map(|(lm, fs)| (lm.fit(&self.measurements.x, &self.measurements.y).1, lm, fs))
            .collect::<Vec<_>>();
        let (best_lm, best_remaining_fs) = potential_models_fit
            .into_iter()
            .filter(|(rmse, _, _)| rmse.is_finite())
            .min_by(|(rmse1, _, _), (rmse2, _, _)| rmse1.total_cmp(rmse2))
            .map(|(_, lm, fs)| (lm, fs))
            .expect("at least one fit yields finite rmse");

        self.linear_model = best_lm.clone();
        self.feature_set = best_remaining_fs.clone();

        let best_model_cardinality = best_lm.n_features();
        let previous_best_lm = self
            .sffs_book
            .insert(best_model_cardinality, best_lm.clone());
        self.sffs_book_trace.push(self.sffs_book.clone());

        let prev_match = previous_best_lm.is_some_and(|pm| pm == best_lm);
        if prev_match || self.sffs_trace_loop_found() {
            match best_model_cardinality == self.options.termination.cardinality {
                true => Step::Termination(TerminationReason::BestModelFound),
                false => self.forward_selection(None),
            }
        } else {
            self.backward_selection()
        }
    }

    fn backward_selection(&mut self) -> Step<D> {
        if self.linear_model.n_features() <= 1 {
            return self.forward_selection(None);
        }
        // remove each feature; if the best removal is the just-added feature, stop backward
        // selection (the forward step should not be undone)
        let potential_models: Vec<_> = (0..self.linear_model.n_features())
            .map(|idx| {
                let mut lm_fs = FeatureSet::from(self.linear_model.clone());
                let removed_feature = lm_fs.remove(idx);
                let mut remaining_fs = self.feature_set.clone();
                remaining_fs.push_shared(removed_feature);
                let lm = LinearModel::from(lm_fs);
                (lm, remaining_fs)
            })
            .filter(|(lm, _)| {
                let (lower_bound, upper_bound) = match self.options.design_constraint {
                    DesignConstraint::Bound(x) => (x.lower(), x.upper()),
                };
                lm.dimension_dependency_check(lower_bound, upper_bound)
            })
            .collect();

        let potential_models_fit: Vec<_> = potential_models
            .into_par_iter()
            .map(|(lm, fs)| (lm.fit(&self.measurements.x, &self.measurements.y).1, lm, fs))
            .collect();

        let (best_lm, best_remaining_fs) = potential_models_fit
            .into_iter()
            .filter(|(rmse, _, _)| rmse.is_finite())
            .min_by(|(rmse1, _, _), (rmse2, _, _)| rmse1.total_cmp(rmse2))
            .map(|(_, lm, fs)| (lm, fs))
            .expect("at least one fit yields finite rmse");

        self.linear_model = best_lm.clone();
        self.feature_set = best_remaining_fs;

        let best_lm_cardinality = best_lm.n_features();
        let prev_best_model = self.sffs_book.insert(best_lm_cardinality, best_lm.clone());
        self.sffs_book_trace.push(self.sffs_book.clone());

        let prev_match = prev_best_model.is_some_and(|pm| pm == best_lm);
        if prev_match || self.sffs_trace_loop_found() {
            self.forward_selection(None)
        } else {
            self.backward_selection()
        }
    }

    fn sffs_trace_loop_found(&self) -> bool {
        let mut count = 0;
        for trace_book in self.sffs_book_trace.iter().rev().skip(1) {
            if self.sffs_book == *trace_book {
                count += 1;
            }
            if count >= SFFS_LOOP_REPEAT_THRESHOLD {
                return true;
            }
        }
        false
    }
}

mod tests {
    use super::*;
    use crate::{Feature, FeatureFunction, Grid};
    use nalgebra::{SVector, Vector2};
    use num_dual::DualNum;

    #[derive(Feature)]
    #[dimension = 2]
    struct Monomial {
        i: i32,
        j: i32,
    }

    impl FeatureFunction<2> for Monomial {
        fn f<D: DualNum<f64>>(&self, x: &SVector<D, 2>) -> D {
            x[0].powi(self.i) * x[1].powi(self.j)
        }
    }

    fn get_feature_set() -> FeatureSet<2> {
        let mut fs = FeatureSet::new();
        for i in 0..2 {
            for j in 0..2 {
                let monomial = Monomial { i, j };
                fs.push(monomial);
            }
        }
        fs
    }

    fn get_design_constraint() -> DesignConstraint<2> {
        let lower = Vector2::new(-1., -1.);
        let upper = Vector2::new(1., 1.);
        DesignConstraint::Bound(DesignBound::new(lower, upper).unwrap())
    }

    #[test]
    fn empty_feature_set() -> Result<()> {
        let feature_set = get_feature_set();
        let null_model = LinearModel::from(feature_set);
        let empty_feature_set = FeatureSet::<2>::new();
        let design_constraint = get_design_constraint();
        let size = 11;
        let q = Vector2::new(size, size);
        let lower = Vector2::new(-1., -1.);
        let upper = Vector2::new(1., 1.);
        let grid = Grid::new(lower, upper, q)?;
        let init_design = Design::new_from_supp(grid.points);
        let termination = Termination::new(2);
        let option = FeatureSelectionOptions::new(design_constraint, 2, init_design, termination);
        let fs = FeatureSelection::new(null_model, empty_feature_set, option);
        assert_eq!(
            fs.err(),
            Some(Error::EmptyArgument {
                name: "feature_set"
            })
        );
        Ok(())
    }

    #[test]
    fn linear_model_missing_dependency() -> Result<()> {
        let mut feature_set = get_feature_set();
        let mut init_feature_set = FeatureSet::new();
        init_feature_set.push_shared(feature_set.remove(0));
        let null_model = LinearModel::from(init_feature_set);
        let design_constraint = get_design_constraint();
        let size = 11;
        let q = Vector2::new(size, size);
        let lower = Vector2::new(-1., -1.);
        let upper = Vector2::new(1., 1.);
        let grid = Grid::new(lower, upper, q)?;
        let init_design = Design::new_from_supp(grid.points);
        let termination = Termination::new(2);
        let option = FeatureSelectionOptions::new(design_constraint, 2, init_design, termination);
        let fs = FeatureSelection::new(null_model, feature_set.into(), option);
        assert_eq!(fs.err(), Some(Error::LinearModelMissingDependency));
        Ok(())
    }

    #[test]
    fn linear_model_cardinality_target_unreachable() -> Result<()> {
        let mut feature_set = get_feature_set();
        let mut init_feature_set = FeatureSet::new();
        for _ in 0..3 {
            init_feature_set.push_shared(feature_set.remove(0));
        }
        let null_model = LinearModel::from(init_feature_set);
        let design_constraint = get_design_constraint();
        let size = 11;
        let q = Vector2::new(size, size);
        let lower = Vector2::new(-1., -1.);
        let upper = Vector2::new(1., 1.);
        let grid = Grid::new(lower, upper, q)?;
        let init_design = Design::new_from_supp(grid.points);
        let termination = Termination::new(10);
        let option =
            FeatureSelectionOptions::new(design_constraint, 2, init_design, termination.clone());
        let fs = FeatureSelection::new(null_model.clone(), feature_set.clone().into(), option);
        assert_eq!(
            fs.err(),
            Some(Error::TargetModelCardinalityUnreachable {
                cardinality_target: termination.cardinality,
                cardinality_null_model: null_model.n_features(),
                cardinality_feature_set: feature_set.len()
            })
        );
        Ok(())
    }
}
