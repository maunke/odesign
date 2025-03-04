# odesign

`odesign` is an optimal design of experiments library written in pure rust.

## Get started

Please have a look at the book on [odesign.rs](https://odesign.rs) for a high
level introduction and theoretical background and at the docs on
[docs.rs/odesign](https://docs.rs/odesign) for the implementation details.

## Basic Example

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

## Mailings Lists

- [odesign-announce](https://lists.sr.ht/~maunke/odesign-announce): Low-volume
  mailing list for announcements
- [odesign-discuss](https://lists.sr.ht/~maunke/odesign-discuss): Mailing list
  for end-user discussion and questions
- [odesign-devel](https://lists.sr.ht/~maunke/odesign-devel): Mailing list for
  development discussion and patches. For help sending patches to this list,
  please consult [git-send-email.io](https://git-send-email.io).
