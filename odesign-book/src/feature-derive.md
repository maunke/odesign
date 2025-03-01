# Feature derive

In order to define a differentiable linear model with its features you just need
to define a set of instantiated structs with the feature derive and implement
their feature function. With help of this design arbitrary linear models can be
defined. See here the simple implementation of the following linear model

$$ f(x) = \beta_0 + \beta_1 \cdot x + \beta_2 \cdot \frac{1}{x}$$

with its coefficient \\(\beta\\) we want to estimate in the context of linear
regression.

```rust
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

/// f(x): 1 + x + 1 / x
fn main() -> Result<()> {
    let mut fs = FeatureSet::new();
    for i in -1..2 {
        let c: Arc<_> = Monomial { i }.into();
        fs.push(c);
    }

    let _lm = LinearModel::new(fs.features);

    Ok(())
}
```
