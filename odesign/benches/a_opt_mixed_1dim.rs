use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::{SVector, Vector1};
use num_dual::DualNum;
use odesign::{
    AOptimality, Feature, FeatureFunction, FeatureSet, LinearModel, OptimalDesign, Result,
};
use std::{sync::Arc, time::Duration};

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
struct Exp {}

impl FeatureFunction<1> for Exp {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 1>) -> D {
        (x[0]).exp().powi(-1)
    }
}

// f(x): 1 + x + x^-1 + exp(-x)
fn mixed_1dim(size: usize) -> Result<()> {
    // define set of features
    let mut fs = FeatureSet::new();
    for i in -1..2 {
        let c: Arc<_> = Monomial { i }.into();
        fs.push(c);
    }
    let c: Arc<_> = Exp {}.into();
    fs.push(c);

    // define linear model with features
    let lm = LinearModel::new(fs.features);

    // define optimality, bound and init design args
    let optimality: Arc<_> = AOptimality::new(lm.into()).into();
    let lower = Vector1::new(0.5);
    let upper = Vector1::new(2.5);
    let q = Vector1::new(size);

    // define Optimal Design resolver
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_bound_args(lower, upper)?
        .with_init_design_grid_args(lower, upper, q)?;

    // find optimal design
    // get design by: let design = od.solve();
    // or: let design = od.design();
    od.solve();

    Ok(())
}

fn benchmark_mixed_1dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("A-Optimality Mixed 1Dim");
    group.sample_size(10).warm_up_time(Duration::from_secs(1));
    for size in (51..201).step_by(50) {
        group.bench_with_input(BenchmarkId::new("Grid size", size), &size, |b, &s| {
            b.iter(|| mixed_1dim(s));
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_mixed_1dim);
criterion_main!(benches);
