use nalgebra::SVector;
use num_dual::DualNum;
use odesign::{Feature, FeatureFunction};

#[derive(Feature)]
#[dimension = "1"]
struct Monomial {
    i: i32,
    j: i32,
}

impl FeatureFunction<2> for Monomial {
    fn f<D: DualNum<f64>>(&self, x: &SVector<D, 2>) -> D {
        x[0].powi(self.i) * x[1].powi(self.j)
    }
}

fn main() {
    Monomial { i: 0, j: 0 };
}
