use ActivationDirection::*;
use crate::common::matrix::Matrix;
use crate::common::nural_network_cfg::{ActivationDirection, NuralNetworkCfg};

pub struct NuralLayer<'a> {
    cfg: &'a NuralNetworkCfg,
    biases: Matrix,
    weights: Matrix,
}

impl NuralLayer<'_> {
    pub fn new(
        cfg: &NuralNetworkCfg,
        input_nodes: usize,
        output_nodes: usize,
    ) -> NuralLayer {
        NuralLayer {
            cfg,
            biases: Matrix::rnd(output_nodes, 1),
            weights: Matrix::rnd(output_nodes, input_nodes),
        }
    }

/*    pub fn load(&mut self, file: &File) -> std::io::Result<()> {}

    pub fn save(&self, file: &mut File) -> std::io::Result<()> {
    }*/

    pub fn backward(
        &mut self,
        input: &[f64],
        gradients: &[f64],
        errors: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let input_mx = Matrix::from(input);
        let errors_mx = Matrix::from(errors).transpose();
        let gradients_mx = Matrix::from(gradients).transpose().dot_mul(&errors_mx);

        self.biases = self.biases.add(&gradients_mx);
        self.weights = self.weights.add(&gradients_mx.mul(&input_mx));

        let errors = self.weights.transpose().mul(&errors_mx).data;
        let gradients = input_mx.apply(self.cfg.activation_func(Backward)).data;
        (gradients, errors)
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let output = self.weights
            .mul(&Matrix::from(input).transpose())
            .add(&self.biases)
            .apply(self.cfg.activation_func(Forward));
        output.data
    }
}
