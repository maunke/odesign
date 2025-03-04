# Introduction

`odesign` is an optimal design of experiments library written in pure rust. It
allows to find optimal designs for (quasi-) linear models considering arbitrary
optimalities.

This book serves as a high level introduction and theoretical background.

Please find more resources here:

- docs on [docs.rs/odesign](https://docs.rs/odesign)
- project on sourcehut [sr.ht/~maunke/odesign](https://sr.ht/~maunke/odesign)
- read-only mirror on github [maunke/odesign](https://github.com/maunke/odesign)

## Use Cases

- Fast calculation of optimal designs of arbitrary linear models with custom
  design bounds and optimalities.
- Research in area of optimal designs; e.g. I am working on a new optimal design
  feature selection algorithm, a mixture of SFFS, D-, C- and
  Measurements-Costs-Optimality, allowing to perform model feature selection and
  measurements alternating.

## Core Features

The library consists of three main features:

1. [Feature derive](feature-derive.md)
1. [Arbitrary optimalities](optimalities.md)
1. [Optimal design solver](solver.md)

## Basic example

In short, this is a
[basic example](https://git.sr.ht/~maunke/odesign/tree/main/item/odesign-examples/examples/basic/main.rs)
of an optimal design of the simple polynomial 1 + x within design bounds [-1,
+1] and 101 equally distributed grid points as an init design.

```rust
use nalgebra::{SVector, Vector1};
use num_dual::DualNum;
use odesign::{
    DOptimality, Feature, FeatureFunction, FeatureSet, LinearModel, OptimalDesign, Result,
};
use std::sync::Arc;

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

// f(x): 1 + x
fn main() -> Result<()> {
    let mut fs = FeatureSet::new();
    let c: Arc<_> = Monomial { i: 0 }.into();
    fs.push(c);
    let c: Arc<_> = Monomial { i: 1 }.into();
    fs.push(c);

    let lm = LinearModel::new(fs.features);

    let optimality: Arc<_> = DOptimality::new(lm.into()).into();
    let lower = Vector1::new(-1.0);
    let upper = Vector1::new(1.0);
    let q = Vector1::new(101);

    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_bound_args(lower, upper)?
        .with_init_design_grid_args(lower, upper, q)?;
    od.solve();

    println!("{od}");

    Ok(())
}
// Output
// ---------- Design ----------
// Weight  Support Vector
// 0.5000  [ -1.0000 ]
// 0.5000  [ +1.0000 ]
// -------- Statistics --------
// Optimality measure: 1.000000
// No. support vectors: 2
// Iterations: 1
// ----------------------------
```
