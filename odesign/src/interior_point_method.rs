use faer::{linalg::solvers::Solve, unzip, zip, Mat};
use faer_ext::{IntoFaer, IntoNalgebra};
use nalgebra::{DMatrix, DVector};
use std::sync::Arc;

/// Interface for functions of which values are minimized by proving value, gradient and hessian
/// methods.
pub trait NLPFunctionTarget {
    /// Returns the value of its function at x.
    fn val(&self, x: &Mat<f64>) -> f64;
    /// Returns the value and gradient of its function at x.
    fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>);
    /// Returns the value, gradient and hessian of its function at x.
    fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>);
}

pub struct VoronoiConstraint {
    x_id: usize,
    diff: Mat<f64>,
    diff_t: Mat<f64>,
    squared_norm_diff: Mat<f64>,
}

impl VoronoiConstraint {
    pub fn new(mut supp: Mat<f64>, x_id: usize) -> Self {
        let supp_x = supp.clone();
        let supp_x_id = supp_x.col(x_id);
        supp.col_iter_mut().for_each(|mut c| {
            c -= supp_x_id;
            c *= -2.;
        });
        let mut supp_col_squared_norm = Mat::<f64>::zeros(supp.ncols(), 1);
        supp_x.col_iter().enumerate().for_each(|(idx, c)| {
            supp_col_squared_norm[(idx, 0)] = c.squared_norm_l2();
        });
        let squared_norm_x = supp_col_squared_norm[(x_id, 0)];
        supp_col_squared_norm.col_mut(0).iter_mut().for_each(|r| {
            *r = squared_norm_x - *r;
        });
        Self {
            x_id,
            diff: supp.clone(),
            diff_t: supp.transpose().to_owned(),
            squared_norm_diff: supp_col_squared_norm,
        }
    }

    fn tmp_f(&self, x: &Mat<f64>) -> Mat<f64> {
        &self.squared_norm_diff - &self.diff_t * x
    }
}

impl NLPFunctionTarget for VoronoiConstraint {
    fn val(&self, x: &Mat<f64>) -> f64 {
        let tmp_f = self.tmp_f(x);
        let mut val = 0.;

        let supp_size = self.diff.shape().1;
        for i in 0..supp_size {
            if i != self.x_id {
                val -= (-tmp_f[(i, 0)]).ln();
            }
        }
        val
    }
    fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
        let tmp_f = self.tmp_f(x);
        let x_size = x.nrows();
        let mut val = 0.;
        let mut grad = Mat::<f64>::zeros(x_size, 1);

        let supp_size = self.diff.ncols();
        let mut grad_col = grad.col_mut(0);
        for i in 0..supp_size {
            if i != self.x_id {
                let tmp = self.diff.col(i) / tmp_f[(i, 0)];
                val -= (-tmp_f[(i, 0)]).ln();
                grad_col -= tmp;
            }
        }
        (val, grad)
    }
    fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
        let tmp_f = self.tmp_f(x);
        let x_size = x.nrows();
        let mut val = 0.;
        let mut grad = Mat::<f64>::zeros(x_size, 1);
        let mut hes = Mat::<f64>::zeros(x_size, x_size);

        let supp_size = self.diff.ncols();
        let mut grad_col = grad.col_mut(0);
        for i in 0..supp_size {
            if i != self.x_id {
                let tmp = self.diff.col(i) / tmp_f[(i, 0)];
                val -= (-tmp_f[(i, 0)]).ln();
                grad_col -= &tmp;
                hes += &tmp * tmp.transpose();
            }
        }
        (val, grad, hes)
    }
}

/// Ensures the consistency of of [NLPFunctionTarget] value, gradient and hessian methods.
#[macro_export]
macro_rules! assert_nlp_target_consistency {
    ($feature:ident, $x:expr) => {
        let val = $feature.val($x);
        let val_grad = $feature.val_grad($x);
        let val_grad_hes = $feature.val_grad_hes($x);
        assert_eq!(val, val_grad.0);
        assert_eq!(val, val_grad_hes.0);
        assert_eq!(val_grad.1, val_grad_hes.1);
    };
}

/// Configuration of [NLPSolver].
pub struct NLPSolverOptions {
    barrier_prec: f64,
    newton_prec: f64,
    barrier_max_iter: u64,
    newton_max_iter: u64,
    backline_max_iter: u64,
    barrier_mu: f64,
    barrier_t0: f64,
    backline_a: f64,
    backline_b: f64,
}

impl Default for NLPSolverOptions {
    fn default() -> Self {
        Self {
            barrier_prec: 1e-8,
            newton_prec: 1e-6,
            barrier_max_iter: 1_000,
            newton_max_iter: 1_000,
            backline_max_iter: 10,
            barrier_mu: 5.,
            barrier_t0: 100.,
            backline_a: 0.2,
            backline_b: 0.1,
        }
    }
}

impl NLPSolverOptions {
    /// Creates a new nlp solver Configuration with its default values.
    pub fn new() -> Self {
        Self::default()
    }
}

pub struct NLPBound {
    pub lower: Mat<f64>,
    pub upper: Mat<f64>,
}

impl NLPBound {
    pub fn new(lower: DVector<f64>, upper: DVector<f64>) -> Self {
        let lower = lower.view_range(.., ..).into_faer().to_owned();
        let upper = upper.view_range(.., ..).into_faer().to_owned();
        Self { lower, upper }
    }
}

#[cfg_attr(doc, katexit::katexit)]
/// Define the linear equality constraint of [NLPSolver] by providing the matrix M of shape k x
/// m, where m is the size of x and k the number of linear independent constraints.
///
/// Linear equality constraint: $M x = \mathcal{const}, M \in \mathbb R^{k \times m}, \forall x \in
/// \mathbb R^m, \mathcal{const} \in \mathbb R^k$
pub struct LinearEqualityConstraint {
    /// Linear equality constraint matrix.
    pub mat: DMatrix<f64>,
}

pub enum InequalityConstraint {
    Voronoi(VoronoiConstraint),
    Custom(Arc<dyn NLPFunctionTarget + Send + Sync>),
    CustomLogBarrier(Arc<dyn NLPFunctionTarget + Send + Sync>),
}

/// All constraint types for [NLPSolver].
pub struct NLPSolverConstraints {
    /// Cubic constraint for x.
    pub bound: Option<NLPBound>,
    /// Linear equality constraint.
    pub lin_equal: Option<LinearEqualityConstraint>,
    /// Linear inequality constraint.
    pub inequal: Option<InequalityConstraint>,
}

struct NLPPreComputation {
    lin_equal_newton_mat: Option<Mat<f64>>,
}

/// Non linear programming solver that minimizes [NLPFunctionTarget] within given [NLPSolverConstraints].
pub struct NLPSolver {
    options: NLPSolverOptions,
    constraints: NLPSolverConstraints,
    func: Arc<dyn NLPFunctionTarget + Send + Sync>,
    pre_computation: NLPPreComputation,
}

impl NLPSolver {
    /// Initialize the NLP solver.
    pub fn new(
        options: NLPSolverOptions,
        constraints: NLPSolverConstraints,
        func: Arc<dyn NLPFunctionTarget + Send + Sync>,
    ) -> Self {
        let pre_computation = NLPSolver::pre_computation(&constraints);
        Self {
            options,
            constraints,
            func,
            pre_computation,
        }
    }

    fn pre_computation(constraints: &NLPSolverConstraints) -> NLPPreComputation {
        let mut lin_equal_newton_mat: Option<Mat<f64>> = None;
        if let Some(lin_equal) = &constraints.lin_equal {
            let (constr_size, x_size) = lin_equal.mat.shape();
            let mat_size = constr_size + x_size;
            let mut m = DMatrix::zeros(mat_size, mat_size);
            m.view_range_mut(x_size..mat_size, 0..x_size)
                .copy_from(&lin_equal.mat);
            m.view_range_mut(0..x_size, x_size..mat_size)
                .copy_from(&lin_equal.mat.transpose());
            lin_equal_newton_mat = Some(m.view_range(.., ..).into_faer().to_owned());
        }
        NLPPreComputation {
            lin_equal_newton_mat,
        }
    }

    /// Returns x that minimizes the [NLPFunctionTarget] within given [NLPSolverConstraints].
    pub fn minimize(&self, x0: DVector<f64>) -> DVector<f64> {
        self.barrier_method(x0.clone())
    }

    fn barrier_method(&self, x: DVector<f64>) -> DVector<f64> {
        let mut x = x.view_range(.., ..).into_faer().to_owned();
        let x_size = x.nrows() as f64;
        let t0 = 200_f64.max((self.options.barrier_t0 * x_size.sqrt()).min(5e3));
        let mut t = t0;
        let mut i = 0;
        while i < self.options.barrier_max_iter && x_size / t >= self.options.barrier_prec {
            i += 1;
            x = self.newton_method(x, t, t0);
            t *= self.options.barrier_mu;
        }
        x.as_ref().into_nalgebra().column(0).into()
    }

    fn newton_method(&self, mut x: Mat<f64>, t: f64, t0: f64) -> Mat<f64> {
        let x_size = x.nrows();
        let iter_barrier = (t / t0 / self.options.barrier_mu) as i32;

        let mut a = match &self.pre_computation.lin_equal_newton_mat {
            Some(lin_equal_newton_mat) => lin_equal_newton_mat.clone(),
            None => Mat::<f64>::zeros(x_size, x_size),
        };
        let dim = a.nrows();
        let mut b = Mat::<f64>::zeros(dim, 1);

        let mut i = 0;
        let mut crit = 0.;
        let mut backline_exceeded: bool = false;

        while i < self.options.newton_max_iter
            && (i == 0
                || crit >= self.options.newton_prec * 1_f64.max(1e4 * 2_f64.powi(-iter_barrier)))
            && !backline_exceeded
        {
            i += 1;
            let (func_val, mut func_grad, mut func_hes) = self.func.val_grad_hes(&x);
            func_grad *= -t;
            func_hes *= t;

            if let Some(bound) = &self.constraints.bound {
                let (bound_grad, bound_hes) = self.log_barrier_bound_grad_hes(&x, bound);
                func_grad -= bound_grad;
                func_hes += bound_hes;
            }
            if let Some(inequal) = &self.constraints.inequal {
                let (inequal_grad, inequal_hes) = self.log_barrier_inequal_grad_hes(&x, inequal);
                func_grad -= inequal_grad;
                func_hes += inequal_hes;
            }

            a.as_mut()
                .submatrix_mut(0, 0, x_size, x_size)
                .copy_from(func_hes);
            b.as_mut()
                .submatrix_mut(0, 0, x_size, 1)
                .copy_from(func_grad);
            let dx_total = a.partial_piv_lu().solve(&b);
            let dx = dx_total.submatrix(0, 0, x_size, 1).to_owned();
            crit = dx.norm_l2();

            self.backline_search(&mut x, dx, &mut backline_exceeded, func_val);
        }
        x
    }

    #[inline(always)]
    fn backline_search(
        &self,
        x: &mut Mat<f64>,
        mut dx: Mat<f64>,
        backline_exceeded: &mut bool,
        old_func_val: f64,
    ) {
        let dx_norm = dx.norm_l2();
        if dx_norm > 1. {
            dx /= dx_norm;
        }
        let mut a = self.options.backline_a;
        let mut iter = 0;
        let mut search = true;
        while iter < self.options.backline_max_iter && search {
            iter += 1;
            let x_tmp = &*x + a * &dx;
            let func_val = self.func.val(&x_tmp);
            if func_val < old_func_val && self.feasibility_check(&x_tmp) {
                *x = x_tmp.clone();
                search = false;
            } else {
                a *= self.options.backline_b;
            }
        }
        if iter == self.options.backline_max_iter {
            *backline_exceeded = true;
        }
    }

    #[inline(always)]
    fn mat_min(&self, x: &Mat<f64>) -> f64 {
        let mut min = f64::INFINITY;
        x.col_iter().for_each(|c| {
            c.iter().for_each(|&v| {
                if v < min {
                    min = v;
                }
            });
        });
        min
    }

    #[inline(always)]
    fn feasibility_check(&self, x: &Mat<f64>) -> bool {
        if let Some(bound) = &self.constraints.bound {
            return !(self.mat_min(&(x - &bound.lower)) < 0.
                || self.mat_min(&(&bound.upper - x)) < 0.);
        }
        true
    }

    fn log_barrier_inequal_grad_hes(
        &self,
        x: &Mat<f64>,
        inequal: &InequalityConstraint,
    ) -> (Mat<f64>, Mat<f64>) {
        match inequal {
            InequalityConstraint::Voronoi(c) => {
                let (_, grad, hes) = c.val_grad_hes(x);
                (grad, hes)
            }
            InequalityConstraint::Custom(f) => {
                let (f_val, f_grad, f_hes) = f.val_grad_hes(x);
                let grad = 1. / f_val * &f_grad;
                let hes = 1. / f_val.powi(2) * &f_grad * f_grad.transpose() + 1. / f_val * f_hes;
                (grad, hes)
            }
            InequalityConstraint::CustomLogBarrier(f) => {
                let (_, grad, hes) = f.val_grad_hes(x);
                (grad, hes)
            }
        }
    }

    #[inline(always)]
    fn log_barrier_bound_grad_hes(&self, x: &Mat<f64>, bound: &NLPBound) -> (Mat<f64>, Mat<f64>) {
        let mut grad = Mat::<f64>::zeros(x.nrows(), 1);
        zip!(&mut grad, x, &bound.lower, &bound.upper)
            .for_each(|unzip!(g, v, l, u)| *g = 1.0 / (*u - *v) + 1.0 / (*l - *v));
        let mut hes = Mat::<f64>::zeros(x.nrows(), x.nrows());
        for j in 0..hes.ncols() {
            for i in 0..hes.nrows() {
                if i == j {
                    let v = 1.0 / (bound.upper[(i, 0)] - x[(i, 0)]).powi(2)
                        + 1.0 / (bound.lower[(i, 0)] - x[(i, 0)]).powi(2);
                    hes[(i, i)] = v;
                }
            }
        }
        (grad, hes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Feature, FeatureFunction, IntoSVector, Result};
    use faer::mat;
    use faer_ext::IntoFaer;
    use nalgebra::SVector;

    struct NLPTargetTest {}

    impl NLPFunctionTarget for NLPTargetTest {
        fn val(&self, x: &Mat<f64>) -> f64 {
            x[(0, 0)].powi(2)
        }

        fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
            (self.val(x), 2. * x)
        }
        fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
            let vg = self.val_grad(x);
            (vg.0, vg.1, mat![[2.]])
        }
    }

    #[derive(Feature)]
    #[dimension = 1]
    struct VoronoiFunction {
        x_i: SVector<f64, 1>,
        x_j: SVector<f64, 1>,
    }

    impl VoronoiFunction {
        pub fn new(x_i: SVector<f64, 1>, x_j: SVector<f64, 1>) -> Self {
            Self { x_i, x_j }
        }
    }

    impl FeatureFunction<1> for VoronoiFunction {
        fn f<D: num_dual::DualNum<f64>>(&self, x: &nalgebra::SVector<D, 1>) -> D {
            (x[0].clone() - self.x_i[0]).powi(2) - (x[0].clone() - self.x_j[0]).powi(2)
        }
    }

    struct VoronoiFunctionTarget {
        vf: VoronoiFunction,
    }

    impl VoronoiFunctionTarget {
        pub fn new(vf: VoronoiFunction) -> Self {
            Self { vf }
        }
    }

    impl NLPFunctionTarget for VoronoiFunctionTarget {
        fn val(&self, x: &Mat<f64>) -> f64 {
            self.vf.val(&x.into_svector())
        }
        fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
            let val_grad = self.vf.val_grad(&x.into_svector());
            (
                val_grad.0,
                val_grad.1.view_range(.., ..).into_faer().to_owned(),
            )
        }
        fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
            let val_grad_hes = self.vf.val_grad_hes(&x.into_svector());
            (
                val_grad_hes.0,
                val_grad_hes.1.view_range(.., ..).into_faer().to_owned(),
                val_grad_hes.2.view_range(.., ..).into_faer().to_owned(),
            )
        }
    }

    #[test]
    fn test_nlp_solver() -> Result<()> {
        let size = 1;
        for i in 0..10 {
            let lower = (i as f64) / 20.;
            let bound = Some(NLPBound::new(
                DVector::from_element(size, lower),
                DVector::from_element(size, 1.),
            ));
            let constraints = NLPSolverConstraints {
                bound,
                lin_equal: None,
                inequal: None,
            };
            let options = NLPSolverOptions::new();

            let nlp_target: Arc<_> = NLPTargetTest {}.into();

            let solver = NLPSolver::new(options, constraints, nlp_target);
            let x0 = DVector::from_vec(vec![0.9]);
            let x_min = solver.minimize(x0);
            assert!(x_min.relative_eq(&DVector::from_vec(vec![lower]), 1e-4, 1e-4));
        }
        Ok(())
    }

    #[test]
    fn test_nlp_solver_start_outside() -> Result<()> {
        let size = 1;
        let lower = 0.;
        let bound = Some(NLPBound::new(
            DVector::from_element(size, lower),
            DVector::from_element(size, 1.),
        ));
        let constraints = NLPSolverConstraints {
            bound,
            lin_equal: None,
            inequal: None,
        };
        let options = NLPSolverOptions::new();

        let nlp_target: Arc<_> = NLPTargetTest {}.into();

        let solver = NLPSolver::new(options, constraints, nlp_target);
        let x0 = DVector::from_vec(vec![2.9]);
        let x_min = solver.minimize(x0.clone());
        assert!(x_min.relative_eq(&x0, 1e-4, 1e-4));
        Ok(())
    }

    #[test]
    fn test_nlp_solver_linear_constr() -> Result<()> {
        let size = 1;
        for i in 0..10 {
            let lower = (i as f64) / 20.;
            let bound = Some(NLPBound::new(
                DVector::from_element(size, lower),
                DVector::from_element(size, 1.),
            ));

            let lin_equal = Some(LinearEqualityConstraint {
                mat: DMatrix::from_vec(1, 1, vec![1.]),
            });
            let constraints = NLPSolverConstraints {
                bound,
                lin_equal,
                inequal: None,
            };
            let options = NLPSolverOptions::new();

            let nlp_target: Arc<_> = NLPTargetTest {}.into();

            let solver = NLPSolver::new(options, constraints, nlp_target);
            let x0 = DVector::from_vec(vec![0.9]);
            let x_min = solver.minimize(x0.clone());
            assert!(x_min.relative_eq(&x0, 1e-4, 1e-4));
        }
        Ok(())
    }

    #[test]
    fn test_nlp_solver_voronoi_constraint() -> Result<()> {
        let size = 1;
        let lower = 0.;
        for i in 0..10 {
            let x0 = DVector::from_vec(vec![0.5]);
            let bound = Some(NLPBound::new(
                DVector::from_element(size, lower),
                DVector::from_element(size, 1.),
            ));
            let voronoi_lower = lower + 0.01 * i as f64;
            let supp = mat![[voronoi_lower, 0.5, 1.]];
            let inequal = Some(InequalityConstraint::Voronoi(VoronoiConstraint::new(
                supp, 1,
            )));
            let constraints = NLPSolverConstraints {
                bound,
                lin_equal: None,
                inequal,
            };
            let options = NLPSolverOptions::new();

            let nlp_target: Arc<_> = NLPTargetTest {}.into();

            let solver = NLPSolver::new(options, constraints, nlp_target);
            let x_min = solver.minimize(x0);
            assert!(x_min.relative_eq(
                &DVector::from_vec(vec![(voronoi_lower + 0.5) / 2.]),
                1e-4,
                1e-4
            ));
        }
        Ok(())
    }

    #[test]
    fn test_nlp_solver_voronoi_constraint_feature() -> Result<()> {
        let supp = mat![[0., 0.5, 1.]];
        let voronoi = VoronoiConstraint::new(supp, 1);
        let x = mat![[0.5]];

        let v = voronoi.val(&x);
        let vg = voronoi.val_grad(&x);
        let vgh = voronoi.val_grad_hes(&x);

        assert!((vgh.0 - v).abs() < 1e-8);
        assert!((vgh.0 - vg.0).abs() < 1e-8);
        assert!((vgh.1 - vg.1).norm_l2() < 1e-8);
        Ok(())
    }

    #[test]
    fn test_nlp_solver_custom_log_barrier() -> Result<()> {
        let supp = mat![[0., 0.5, 1.]];
        let voronoi = VoronoiConstraint::new(supp, 1);
        let x = mat![[0.5]];

        let vgh = voronoi.val_grad_hes(&x);
        let custom_log_barrier = InequalityConstraint::CustomLogBarrier(Arc::new(voronoi));

        let nlp_target: Arc<_> = NLPTargetTest {}.into();
        let solver = NLPSolver::new(
            NLPSolverOptions::new(),
            NLPSolverConstraints {
                bound: None,
                lin_equal: None,
                inequal: None,
            },
            nlp_target,
        );

        let gh_custom = solver.log_barrier_inequal_grad_hes(&x, &custom_log_barrier);

        assert!((vgh.1 - gh_custom.0).norm_l2() < 1e-8);
        assert!((vgh.2 - gh_custom.1).norm_l2() < 1e-8);
        Ok(())
    }

    #[test]
    fn test_nlp_solver_custom_inequal_constraint() -> Result<()> {
        let supp = mat![[0., 0.5, 1.]];
        let voronoi = VoronoiConstraint::new(supp.clone(), 1);
        let z = mat![[0.4]];
        let vgh = voronoi.val_grad_hes(&z);

        let mut gradient = Mat::<f64>::zeros(1, 1);
        let mut hessian = Mat::<f64>::zeros(1, 1);

        let nlp_target: Arc<_> = NLPTargetTest {}.into();
        let solver = NLPSolver::new(
            NLPSolverOptions::new(),
            NLPSolverConstraints {
                bound: None,
                lin_equal: None,
                inequal: None,
            },
            nlp_target,
        );

        supp.row(0).iter().enumerate().for_each(|(x_id, x)| {
            if x_id != 1 {
                let x_i = SVector::<f64, 1>::from_vec(vec![0.5]);
                let x_j = SVector::<f64, 1>::from_vec(vec![*x]);
                let vf = VoronoiFunction::new(x_i, x_j);
                let vf_target = VoronoiFunctionTarget::new(vf);
                assert!((vf_target.val(&z) - vf_target.val_grad(&z).0).abs() < 1e-8);
                let inequal = InequalityConstraint::Custom(Arc::new(vf_target));
                let inequal_gh = solver.log_barrier_inequal_grad_hes(&z, &inequal);
                gradient += inequal_gh.0;
                hessian += inequal_gh.1;
            }
        });

        assert!((vgh.1 - gradient).norm_l2() < 1e-8);
        assert!((vgh.2 - hessian).norm_l2() < 1e-8);

        Ok(())
    }
}
