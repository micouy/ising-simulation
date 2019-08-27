#![allow(non_snake_case)]

use chrono::prelude::*;
use ndarray::prelude::*;
use pbr::ProgressBar;
use rand::{distributions::Normal, prelude::*};
use rayon::prelude::*;
use serde_json::{json, to_string_pretty};
use indoc::indoc;

use std::{
    env::args,
    fs,
    path::Path,
    sync::mpsc::{channel, Sender},
    thread,
};

use ising_lib::prelude::*;

// loop params
const FLIPS_TO_SKIP: usize = 10_000;
const ATTEMPTS_PER_FLIP: usize = 20;
const FLIPS_PER_MEASUREMENT: usize = 250;
const MEASUREMENTS_PER_STEP: usize = 1000;
const STEPS: usize = 100;

// simulation params
const MEAN_STEPS_BEFORE_SLEEP: usize = 10;
const A: usize = 75;
const POWER: i32 = 3;
const TEMPERATURE: f64 = 2.50;
const SIZE: usize = 50;

struct Params {
    // temperature
    T: f64,

    // alpha param
    alpha: f64,

    // how many steps passes before node's magnetic field turns off
    steps_before_sleep: (usize, usize),

    // how many steps should be skipped to let the lattice stabilize
    flips_to_skip: usize,

    // how many measurements to take at each step
    measurements_per_step: usize,

    // how many flips to make between the measurements
    flips_per_measurement: usize,

    // how many times to attempt to flip the spin before skipping it
    attempts_per_flip: usize,

    lattice_size: usize,

    // how many steps to make
    steps: usize,
}

impl Params {
    fn new(alpha: f64, steps_before_sleep: (usize, usize), T: f64) -> Self {
        assert!(0.0 <= alpha, alpha <= 1.0);

        Self {
            T,
            alpha,
            steps_before_sleep,

            flips_to_skip: FLIPS_TO_SKIP,
            measurements_per_step: MEASUREMENTS_PER_STEP,
            flips_per_measurement: FLIPS_PER_MEASUREMENT,
            attempts_per_flip: ATTEMPTS_PER_FLIP,
            lattice_size: SIZE,
            steps: STEPS,
        }
    }
}

struct Record {
    step: usize,
    dE: f64,
    I: f64,
    X: f64,
}

fn compose_results(records: &[Record], params: Params) -> String {
    let records = records
        .iter()
        .map(|r| {
            json!({
                "I": r.I,
                "dE": r.dE,
                "X": r.X,
            })
        })
        .collect::<Vec<_>>();

    to_string_pretty(&json!({
        "records": records,
        "params": {
            "T": params.T,
            "alpha": params.alpha,
            "stepsBeforeSleep": {
                "mean": params.steps_before_sleep.0,
                "deviation": params.steps_before_sleep.1,
            },
        },
    }))
    .unwrap()
}

fn compose_file_name() -> String {
    let now = Local::now().format("%d.%m.%Y-%H.%M").to_string();
    let id = thread_rng().gen_range(100_i32, 999_i32);

    format!("results-{}-{}.txt", now, id)
}

fn run(params: Params, pb_tx: Sender<()>) -> (String, String) {
    let mut rng = SmallRng::from_entropy();
    let distr = Normal::new(
        params.steps_before_sleep.0 as f64,
        params.steps_before_sleep.1 as f64,
    );
    let mut distr_sample = rng.clone().sample_iter(&distr);
    let mut lattice = Lattice::new((params.lattice_size, params.lattice_size));

    let h = Array2::from_shape_fn(
        (params.lattice_size, params.lattice_size),
        |_| distr_sample.next().unwrap() as usize,
    );

    let map_activity_to_h = |h: &Array2<usize>, step| {
        h.map(|steps_before_sleep| {
            // is the node still active?
            if *steps_before_sleep >= step {
                1.0
            } else if step - *steps_before_sleep <= A {
                1.0 - ((step - *steps_before_sleep) as f64 / A as f64)
                    .powi(POWER)
            } else {
                0.0
            }
        })
    };

    let measure_E_diff = |lattice: &Lattice<_>, ix, h: &Array2<f64>| -> f64 {
        2.0 * f64::from(lattice.inner()[ix])
            * (params.alpha * lattice.measure_I()
                + (1.0 - params.alpha) * h[ix])
    };

    let mut attempt_flip = |lattice: &mut Lattice<_>, h, step| -> bool {
        let ix = lattice.gen_random_index();
        let h = map_activity_to_h(h, step);
        let E_diff = measure_E_diff(&lattice, ix, &h);
        let probability = calc_flip_probability(E_diff, params.T);

        if probability > rng.gen() {
            lattice.flip_spin(ix);

            true // the flip has already occured
        } else {
            false // the flip has not occured yet
        }
    };

    // "cool" the lattice to its natural state
    (0..params.flips_to_skip).for_each(|_| {
        (0..params.attempts_per_flip)
            .map(|_| attempt_flip(&mut lattice, &h, 0))
            .take_while(|already_flipped| !already_flipped)
            .for_each(|_| ());
    });

    let mut records: Vec<Record> = (0..params.steps)
        .map(|step| {
            let (Es, Is) = (0..params.measurements_per_step)
                .map(|_| {
                    (0..params.flips_per_measurement).for_each(|_| {
                        (0..params.attempts_per_flip)
                            .map(|_| attempt_flip(&mut lattice, &h, step))
                            .take_while(|already_flipped| !already_flipped)
                            .for_each(|_| ()); // evaluate the iterator
                    });

                    let _ = pb_tx.send(());

                    (lattice.measure_E(), lattice.measure_I())
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();

            let dE = calc_dE(&Es, params.T);
            let I = calc_I(&Is);
            let X = calc_X(&Es);

            // println!("{}", [dE.to_string(), I.to_string(),
            // X.to_string()].connect(","));

            Record { step, dE, I, X }
        })
        .collect();

    let file_name = compose_file_name();
    records.sort_by_key(|r| r.step);
    let results = compose_results(&records, params);

    (results, file_name)
}

fn main() {
    let dir_name = match args().nth(1) {
        None => {
            println!("{}", indoc!("
                Ising Model simulation

                USAGE:
                    cmd <DIR>

                ARGS:
                    <DIR> - where to save the results to
            "));

            return;
        }
        Some(dir_name) => dir_name,
    };

    // make sure it's a valid directory
    assert!(Path::new(&dir_name).is_dir());

    let deviations = vec![8];
    let alphas = vec![0.8];

    // TEST
    // let deviations = vec![5_usize];
    // let alphas = vec![0.15_f64];

    let params = alphas
        .into_iter()
        .map(|alpha| {
            deviations.iter().map(move |deviation| (alpha, *deviation))
        })
        .flatten()
        .collect::<Vec<_>>();

    let bar_count =
        STEPS as u64 * MEASUREMENTS_PER_STEP as u64 * params.len() as u64;
    let (pb_tx, pb_rx) = channel();

    let handle = thread::spawn(move || {
        let mut pb = ProgressBar::new(bar_count);
        pb.set_width(Some(100));
        pb.show_message = true;
        pb.message("Running...");

        for _ in 0..bar_count {
            let _ = pb_rx.recv();
            pb.inc();
        }

        pb.finish_print("Finished!");
    });

    let results = params
        .into_iter()
        .zip((0..1).map(|_| pb_tx.clone()))
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|((alpha, deviation), pb_tx)| {
            let params = Params::new(
                alpha,
                (MEAN_STEPS_BEFORE_SLEEP, deviation),
                TEMPERATURE,
            );

            run(params, pb_tx)
        })
        .collect::<Vec<(String, String)>>();

    let _ = handle.join();

    for (result, file_name) in results {
        let path = format!("{}/{}", dir_name, file_name);
        fs::write(path, result.as_bytes()).unwrap();
    }
}
