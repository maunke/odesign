use faer::Mat;
use nalgebra::{SVector, Vector1};
use num_dual::DualNum;
use odesign::{
    DOptimality, Feature, FeatureFunction, FeatureSet, LinearModel, MatrixDRows, NLPFunctionTarget,
    OptimalDesign, Optimalities, Optimality, Result,
};
use std::sync::Arc;

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

// costs optimality (searching for argmax): exp(-sum w * ||x||^2) = measure
// thus minimize in solver: sum w ||x||^2 = value
// sum over all design points (weight, vector)
struct MeasurementCosts<const D: usize> {}

impl<const D: usize> Optimality<D> for MeasurementCosts<D> {
    fn measure(&self, weights: &Mat<f64>, supp: Arc<odesign::MatrixDRows<D>>) -> f64 {
        let m_mean = self.matrix_mean(supp);
        let val = m_mean.val(weights);
        (-val).exp()
    }

    fn matrix_mean(
        &self,
        supp: Arc<odesign::MatrixDRows<D>>,
    ) -> Arc<dyn NLPFunctionTarget + Send + Sync> {
        Arc::new(CostsMatrixMean::new(supp))
    }

    fn dispersion_function(
        &self,
        _supp: Arc<odesign::MatrixDRows<D>>,
        weights: Mat<f64>,
        x_id: usize,
    ) -> Arc<dyn NLPFunctionTarget + Send + Sync> {
        Arc::new(CostsDispersionFunction::new(weights, x_id, D))
    }
}

struct CostsMatrixMean<const D: usize> {
    supp_norm_l2: Mat<f64>,
}

impl<const D: usize> CostsMatrixMean<D> {
    fn new(supp: Arc<MatrixDRows<D>>) -> Self {
        let mut supp_norm_l2 = Mat::<f64>::zeros(supp.ncols(), 1);
        supp.column_iter().enumerate().for_each(|(idx, c)| {
            let value = c.norm_squared();
            supp_norm_l2[(idx, 0)] = value;
        });
        Self { supp_norm_l2 }
    }
}

impl<const D: usize> NLPFunctionTarget for CostsMatrixMean<D> {
    fn val(&self, x: &Mat<f64>) -> f64 {
        x.col(0).transpose() * self.supp_norm_l2.col(0)
    }

    fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
        let val = self.val(x);
        let grad = self.supp_norm_l2.clone();
        (val, grad)
    }

    fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
        let (val, grad) = self.val_grad(x);
        let hes = Mat::<f64>::zeros(x.nrows(), x.nrows());
        (val, grad, hes)
    }
}

struct CostsDispersionFunction {
    weight: f64,
    weight_factor: f64,
    weight_factor_mat: Mat<f64>,
}

impl CostsDispersionFunction {
    fn new(weights: Mat<f64>, x_id: usize, dim: usize) -> Self {
        let weight = weights[(x_id, 0)];
        let weight_factor = 2. * weight;
        let mut weight_factor_mat = Mat::<f64>::zeros(dim, dim);
        for i in 0..dim {
            weight_factor_mat[(i, i)] = weight_factor;
        }
        Self {
            weight,
            weight_factor,
            weight_factor_mat,
        }
    }
}

impl NLPFunctionTarget for CostsDispersionFunction {
    fn val(&self, x: &Mat<f64>) -> f64 {
        self.weight * x.as_ref().squared_norm_l2()
    }

    fn val_grad(&self, x: &Mat<f64>) -> (f64, Mat<f64>) {
        let val = self.val(x);
        let grad = self.weight_factor * x;
        (val, grad)
    }

    fn val_grad_hes(&self, x: &Mat<f64>) -> (f64, Mat<f64>, Mat<f64>) {
        let (val, grad) = self.val_grad(x);
        let hes = self.weight_factor_mat.clone();
        (val, grad, hes)
    }
}

// f(x): 1 + x
fn main() -> Result<()> {
    // define set of features
    let mut fs = FeatureSet::new();
    for i in 0..4 {
        let c: Arc<_> = Monomial { i }.into();
        fs.push(c);
    }

    // define linear model with features
    let lm = LinearModel::new(fs.features);

    // define optimality, bound and init design args
    let d_optimality: Arc<_> = DOptimality::new(lm.into()).into();
    let costs_optimality: Arc<_> = MeasurementCosts {}.into();
    let optimalities: Optimalities<1> = vec![d_optimality, costs_optimality];
    let optimalities_weights = vec![1., 10.];
    let lower = Vector1::new(-1.0);
    let upper = Vector1::new(1.0);
    let q: SVector<usize, 1> = Vector1::new(11);

    // define Optimal Design resolver
    let mut od = OptimalDesign::new()
        .with_optimalities(optimalities, optimalities_weights)
        .with_bound_args(lower, upper)?
        .with_init_design_grid_args(lower, upper, q)?;

    od.solve();

    // display optimal design
    println!("{od}");

    Ok(())
}
