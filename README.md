# odesign

`odesign` is an optimal design of experiments library written in pure rust.
There are at least these following use cases:

- Fast calculation of optimal designs of arbitrary linear models with custom
  design bounds and optimalities.
- Research in area of optimal designs; e.g. I am working on a new optimal design
  feature selection algorithm, a mixture of SFFS, D-, C- and
  Measurements-Costs-Optimality, allowing to perform model feature selection and
  measurements alternating.

## Get started

Please have a look at the book on [odesign.rs](https://odesign.rs) for a high
level introduction and theoretical background and at the docs on
[docs.rs/odesign](https://docs.rs/odesign) for the implementation details.

## Community

### Mailing lists

- [odesign-announce](https://lists.sr.ht/~maunke/odesign-announce): Low-volume
  mailing list for announcements
- [odesign-discuss](https://lists.sr.ht/~maunke/odesign-discuss): Mailing list
  for end-user discussion and questions
- [odesign-devel](https://lists.sr.ht/~maunke/odesign-devel): Mailing list for
  development discussion and patches. For help sending patches to this list,
  please consult [git-send-email.io](https://git-send-email.io).

### Tickets

The tracker on [todo.sr.ht/~maunke/odesign](https://todo.sr.ht/~maunke/odesign)
is for confirmed bugs and confirmed feature requests only.

Before creating a ticket, search for existing (possibly already fixed) issues,
on the docs or in the mailing list archives: odesign-discuss, odesign-devel.

If you cannot find anything describing your issue or if you have a question, ask
on one of the the mailing lists first. You will be asked to file a ticket if
appropriate.

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

## Advanced Example

As a more complex example please take a look at this [3 dimensional polynomial example](https://git.sr.ht/~maunke/odesign/tree/main/item/odesign-examples/examples/polynomial-3dim/main.rs) with monoms lower than a degree of 3 within a
design space of [-1, -1, -1] x [+1, +1, +1] and an initial design grid of 11 x 11 x 11 points.

```rust
use nalgebra::{SVector, Vector3};
use num_dual::DualNum;
use odesign::{
    DOptimality, Feature, FeatureFunction, FeatureSet, LinearModel, OptimalDesign, Result,
};
use std::sync::Arc;

#[derive(Feature)]
#[dimension = 3]
struct Monomial {
    i: i32,
    j: i32,
    k: i32,
}

impl FeatureFunction<3> for Monomial {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 3>) -> D {
        x[0].powi(self.i) * x[1].powi(self.j) * x[2].powi(self.k)
    }
}

// f(x, y, z):  1 + x + y + z
//              + x * y + x * z + y * y
//              + x ^ 2 + y ^ 2 + z ^ 2
fn main() -> Result<()> {
    let mut fs = FeatureSet::new();
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                if i + j + k < 3 {
                    let c: Arc<_> = Monomial { i, j, k }.into();
                    fs.push(c);
                }
            }
        }
    }

    let lm = LinearModel::new(fs.features);

    let q = Vector3::new(11, 11, 11);
    let lower = Vector3::new(-1., -1., -1.);
    let upper = Vector3::new(1., 1., 1.);
    let optimality = Arc::new(DOptimality::new(lm.into()));
    let mut od = OptimalDesign::new()
        .with_optimality(optimality)
        .with_bound_args(lower, upper)?
        .with_init_design_grid_args(lower, upper, q)?;
    od.solve();

    println!("{od}");

    Ok(())
}
// Output
// -------------- Design ---------------
// Weight	Support Vector
// 0.0689	[ -1.0000, -1.0000, -1.0000 ]
// 0.0251	[ -1.0000, -1.0000, +0.0000 ]
// 0.0689	[ -1.0000, -1.0000, +1.0000 ]
// 0.0251	[ -1.0000, +0.0000, -1.0000 ]
// 0.0205	[ -1.0000, +0.0000, +0.0000 ]
// 0.0251	[ -1.0000, +0.0000, +1.0000 ]
// 0.0689	[ -1.0000, +1.0000, -1.0000 ]
// 0.0251	[ -1.0000, +1.0000, +0.0000 ]
// 0.0689	[ -1.0000, +1.0000, +1.0000 ]
// 0.0251	[ +0.0000, -1.0000, -1.0000 ]
// 0.0205	[ +0.0000, -1.0000, +0.0000 ]
// 0.0251	[ +0.0000, -1.0000, +1.0000 ]
// 0.0205	[ +0.0000, +0.0000, -1.0000 ]
// 0.0245	[ +0.0000, +0.0000, +0.0000 ]
// 0.0205	[ +0.0000, +0.0000, +1.0000 ]
// 0.0251	[ +0.0000, +1.0000, -1.0000 ]
// 0.0205	[ +0.0000, +1.0000, +0.0000 ]
// 0.0251	[ +0.0000, +1.0000, +1.0000 ]
// 0.0689	[ +1.0000, -1.0000, -1.0000 ]
// 0.0251	[ +1.0000, -1.0000, +0.0000 ]
// 0.0689	[ +1.0000, -1.0000, +1.0000 ]
// 0.0251	[ +1.0000, +0.0000, -1.0000 ]
// 0.0205	[ +1.0000, +0.0000, +0.0000 ]
// 0.0251	[ +1.0000, +0.0000, +1.0000 ]
// 0.0689	[ +1.0000, +1.0000, -1.0000 ]
// 0.0251	[ +1.0000, +1.0000, +0.0000 ]
// 0.0689	[ +1.0000, +1.0000, +1.0000 ]
// ------------ Statistics -------------
// Optimality measure: 0.474478
// No. support vectors: 27
// Iterations: 2
// -------------------------------------
```

## Roadmap

- Optimalities:
  - [x] D-Optimality
  - [x] C-Optimality
  - [x] Custom Optimality
  - [x] A-Optimality
  - [x] Costs Optimality
- Design Bounds:
  - [x] Cubic Bounds
  - [x] Custom Bounds
- Documentation:
  - [ ] documentation of the optimal design solver backed by "adaptive grid
        semidefinite programming for finding optimal designs" (doi:
        [10.1007/s11222-017-9741-y](https://doi.org/10.1007/s11222-017-9741-y))
- Research:
  - [ ] New optimal design feature selection algorithm, a mixture of SFFS, D-,
        C- and Measurements-Costs-Optimality, allowing to perform model feature
        selection and measurements alternating
