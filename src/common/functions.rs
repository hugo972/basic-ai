use std::f64::consts::E;

pub struct Func<'a> {
    pub dx: &'a dyn Fn(f64) -> f64,
    pub fx: &'a dyn Fn(f64) -> f64,
}

pub const RE_LU: Func = Func {
    dx: &|x| x,
    fx: &|x| f64::max(0.0, x),
};

pub const SIGMOID: Func = Func {
    dx: &|x| x * (1.0 - x),
    fx: &|x| 1.0 / (E.powf(-x) + 1.0),
};

#[allow(dead_code)]
pub fn softmax(vec: &mut [f64]) {
    let mut sum = 0.0;
    for value in vec.into_iter() {
        *value = E.powf(*value);
        sum += *value;
    }

    for value in vec.into_iter() {
        *value /= sum;
    }
}
