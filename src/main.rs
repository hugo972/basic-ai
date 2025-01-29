mod common;

use crate::common::nural_network::NuralNetwork;
use crate::common::nural_network_cfg::NuralNetworkCfg;

fn main() {
    let cfg = NuralNetworkCfg {
        layer_sizes: [2, 3, 1].to_vec(),
        ..Default::default()
    };
    let mut nural_network = NuralNetwork::new(&cfg);

    nural_network.train(
        &[
            (&[0.0, 0.0], &[0.0]),
            (&[0.0, 1.0], &[1.0]),
            (&[1.0, 0.0], &[1.0]),
            (&[1.0, 1.0], &[0.0]),
        ],
        1000,
    );

    println!("trained results:");
    println!("0 xor 0 = {:?}", nural_network.query(&[0.0, 0.0]));
    println!("0 xor 1 = {:?}", nural_network.query(&[0.0, 1.0]));
    println!("1 xor 0 = {:?}", nural_network.query(&[1.0, 0.0]));
    println!("1 xor 1 = {:?}", nural_network.query(&[1.0, 1.0]));
}
