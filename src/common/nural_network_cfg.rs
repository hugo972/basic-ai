use crate::common::functions::{RE_LU, SIGMOID};

#[derive(Default)]
pub struct NuralNetworkCfg {
    pub activation: NuralNetworkCfgActivation,
    pub debug: bool,
    pub layer_sizes: Vec<usize>,
}

#[allow(dead_code)]
#[derive(Default)]
pub enum NuralNetworkCfgActivation {
    ReLu,
    #[default]
    Sigmoid,
}

pub enum ActivationDirection {
    Backward,
    Forward,
}

impl NuralNetworkCfg {
    pub fn activation_func(&self, dir: ActivationDirection) -> impl Fn(f64, usize) -> f64 {
        let func = match self.activation {
            NuralNetworkCfgActivation::ReLu => RE_LU,
            NuralNetworkCfgActivation::Sigmoid => SIGMOID,
        };

        let func_closure = match dir {
            ActivationDirection::Backward => func.dx,
            ActivationDirection::Forward => func.fx,
        };

        |val, _| func_closure(val)
    }
}
