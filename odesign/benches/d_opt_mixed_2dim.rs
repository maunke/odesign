use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::{SVector, Vector2};
use num_dual::DualNum;
use odesign::{
    DOptimality, Feature, FeatureFunction, FeatureSet, LinearModel, OptimalDesign, Result,
};
use std::{sync::Arc, time::Duration};

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

#[derive(Feature)]
#[dimension = 2]
struct ExpX {}

impl FeatureFunction<2> for ExpX {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 2>) -> D {
        (x[0]).exp().powi(-1)
    }
}

#[derive(Feature)]
#[dimension = 2]
struct ExpY {}

impl FeatureFunction<2> for ExpY {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 2>) -> D {
        (x[1]).exp().powi(-1)
    }
}

#[derive(Feature)]
#[dimension = 2]
struct ExpXY {}

impl FeatureFunction<2> for ExpXY {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 2>) -> D {
        (-x.product()).exp()
    }
}

// f(x, y): 1 + x + y + exp(-x) + exp(-y) + exp[-x*y]
fn mixed_2dim(size: usize) -> Result<()> {
    let mut fs = FeatureSet::new();
    let c: Arc<_> = Monomial { i: 0, j: 0 }.into();
    fs.push(c);
    let c: Arc<_> = Monomial { i: 1, j: 0 }.into();
    fs.push(c);
    let c: Arc<_> = Monomial { i: 0, j: 1 }.into();
    fs.push(c);
    let c: Arc<_> = ExpX {}.into();
    fs.push(c);
    let c: Arc<_> = ExpY {}.into();
    fs.push(c);
    let c: Arc<_> = ExpXY {}.into();
    fs.push(c);

    let lm = LinearModel::new(fs.features);

    let q = Vector2::new(size, size);
    let lower = Vector2::new(-1., -1.);
    let upper = Vector2::new(1., 1.);
    let optimality = Arc::new(DOptimality::new(lm.into()));
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_bound_args(lower, upper)?
        .with_init_design_grid_args(lower, upper, q)?;
    od.solve();

    Ok(())
}

fn benchmark_mixed_2dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("D-Optimality Mixed 2Dim");
    group.sample_size(10).warm_up_time(Duration::from_secs(1));
    for size in (11..41).step_by(10) {
        group.bench_with_input(BenchmarkId::new("Grid size", size), &size, |b, &s| {
            b.iter(|| mixed_2dim(s));
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_mixed_2dim);
criterion_main!(benches);
