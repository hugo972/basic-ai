use crate::common::matrix::Matrix;
use crate::common::nural_layer::NuralLayer;
use crate::common::nural_network_cfg::{ActivationDirection, NuralNetworkCfg};
use ActivationDirection::Backward;

pub struct NuralNetwork<'a> {
    cfg: &'a NuralNetworkCfg,
    layers: Vec<NuralLayer<'a>>,
}

#[derive(Clone, Debug)]
pub struct NuralNetworkResult {
    pub data: Vec<Vec<f64>>,
}

impl NuralNetwork<'_> {
    pub fn new(cfg: &NuralNetworkCfg) -> NuralNetwork {
        NuralNetwork {
            cfg,
            layers: cfg
                .layer_sizes
                .iter()
                .take(cfg.layer_sizes.len() - 1)
                .zip(cfg.layer_sizes.iter().skip(1))
                .map(|(inputs, outputs)| NuralLayer::new(&cfg, *inputs, *outputs))
                .collect(),
        }
    }

    /*    pub fn load(&mut self, filename: &str) {

        }

        pub fn save(&self, filename: &str) -> std::io::Result<()> {
            let mut file = File::create("foo.txt")?;

            for layer in &self.layers {
                layer.save(&mut file)?;
            }

            Ok(())
        }
    */
    pub fn query(&self, input: &[f64]) -> Vec<f64> {
        self.forward(input).output()
    }

    pub fn train(&mut self, data: &[(&[f64], &[f64])], epochs: usize) {
        for epoch in 0..epochs {
            if epoch % 100 == 0 && self.cfg.debug {
                println!("Training epoch {}/{}", epoch + 1, epochs);
            }

            for (input, output) in data.iter() {
                let result = self.forward(input);
                self.backward(&result, output);

                if epoch % 100 == 0 && self.cfg.debug {
                    let max_layer_size = self.cfg.layer_sizes.iter().max().unwrap();
                    for (layer_index, layer_size) in self.cfg.layer_sizes.iter().enumerate() {
                        print!("{}", " ".repeat((max_layer_size - layer_size) / 2 * 6));

                        for value in &result.data[layer_index] {
                            print!("{:.3} ", *value);
                        }

                        println!();
                    }

                    println!("{}", "-".repeat(max_layer_size * 6));
                }
            }
        }
    }

    fn backward(&mut self, result: &NuralNetworkResult, target: &[f64]) {
        let output = Matrix::from(result.data.last().unwrap());
        let mut errors = Matrix::from(target).sub(&output).data;
        let mut gradients = output.apply(self.cfg.activation_func(Backward)).data;

        for (layer, input) in &mut self
            .layers
            .iter_mut()
            .rev()
            .zip(result.data.iter().rev().skip(1))
        {
            (gradients, errors) = layer.backward(input, &gradients, &errors);
        }
    }

    fn forward(&self, input: &[f64]) -> NuralNetworkResult {
        let mut result = NuralNetworkResult {
            data: vec![input.to_vec(); 1],
        };

        for layer in &self.layers {
            let output = layer.forward(result.data.last().unwrap());
            result.data.push(output);
        }

        result
    }
}

impl NuralNetworkResult {
    pub fn output(&self) -> Vec<f64> {
        self.data.last().unwrap().clone()
    }
}
