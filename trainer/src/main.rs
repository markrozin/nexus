//! Train the nexus NNUE with bullet.
//!
//! Training needs a GPU. bullet compiles with no backend enabled, but that
//! build runs on a mock runtime that panics at the first gradient, so a local
//! `cargo build` proves only that this file matches the bullet API.
//!
//! The real run, on a rented NVIDIA GPU with the CUDA toolkit installed and
//! `CUDA_PATH` pointing at it:
//!
//! ```text
//! cargo run --release --features cuda -- --data ../data/selfplay.data --superbatches 40
//! ```
//!
//! A smoke run on a small verified file first, to prove the pipeline end to end
//! before paying for the real one:
//!
//! ```text
//! cargo run --release --features cuda -- --data ../data/tp8.shuffled.data --batch-size 1024 --batches-per-superbatch 16 --superbatches 4 --lr-step 3
//! ```
//!
//! # The contract
//!
//! Architecture, quantisation and input features here are a contract with
//! `src/nnue.rs` in the engine. Change one side and the other must change with
//! it, or the engine silently misreads the network: evaluations come out wrong
//! by a consistent factor or sign rather than failing to load. The engine's
//! `netcheck` binary exists to catch exactly that.
//!
//! Adapted from bullet `examples/simple.rs` at the pinned commit; the graph, save
//! format and loss are unchanged from it.

use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};

/// Must equal `nexus::nnue::HIDDEN`.
const HIDDEN_SIZE: usize = 128;
/// Must equal `nexus::nnue::{SCALE, QA, QB}`.
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

struct Args {
    data: String,
    out: String,
    net_id: String,
    superbatches: usize,
    batches_per_superbatch: usize,
    batch_size: usize,
    threads: usize,
    wdl: f32,
    lr: f32,
    lr_step: usize,
    save_rate: usize,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        raw.iter()
            .position(|a| a == flag)
            .and_then(|i| raw.get(i + 1))
            .cloned()
    };
    let num = |flag: &str, default: usize| -> usize {
        get(flag).and_then(|v| v.parse().ok()).unwrap_or(default)
    };
    let float = |flag: &str, default: f32| -> f32 {
        get(flag).and_then(|v| v.parse().ok()).unwrap_or(default)
    };

    let Some(data) = get("--data") else {
        eprintln!(
            "usage: trainer --data <file.data> [--out nets] [--id name] \
             [--superbatches 40] [--batches-per-superbatch 6104] [--batch-size 16384] \
             [--threads 4] [--wdl 0.4] [--lr 0.001] [--lr-step N] [--save-rate N]"
        );
        std::process::exit(2);
    };

    let superbatches = num("--superbatches", 40).max(1);
    Args {
        data,
        out: get("--out").unwrap_or_else(|| "nets".to_string()),
        net_id: get("--id").unwrap_or_else(|| "nexus".to_string()),
        superbatches,
        // bullet's default: ~100M positions per superbatch at batch 16384.
        batches_per_superbatch: num("--batches-per-superbatch", 6104).max(1),
        batch_size: num("--batch-size", 16_384).max(1),
        threads: num("--threads", 4).max(1),
        // Weight on the game result; the rest goes on the search score.
        //
        // bullet's example uses 0.75, for its own data. Ours comes from a
        // weaker engine whose game results are noisy -- in a sample, only about
        // 80% of positions the search called decisive went on to that result --
        // so this leans on the search score instead. A starting choice, not a
        // tuned one.
        wdl: float("--wdl", 0.4),
        lr: float("--lr", 0.001),
        // Cut the learning rate 10x every this many superbatches. bullet StepLR
        // reads `step` as a period, not a position: its example uses 18 of 40,
        // which cuts at 18 and again at 36. This flag was once named as if it
        // were a single drop point, and a short smoke run printed "drop every
        // 1 superbatches" -- cutting the rate tenfold every superbatch.
        lr_step: num("--lr-step", (superbatches * 45 / 100).max(1)),
        save_rate: num("--save-rate", (superbatches / 4).max(1)),
    }
}

fn main() {
    let args = parse_args();
    println!(
        "training {}x2 net on {}: {} superbatches of {} batches x {}, wdl {}, lr {} cut 10x every {} superbatches",
        HIDDEN_SIZE,
        args.data,
        args.superbatches,
        args.batches_per_superbatch,
        args.batch_size,
        args.wdl,
        args.lr,
        args.lr_step
    );

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 768, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, 1);

            // Side to move first: that ordering is what tells the network whose
            // turn it is, and the engine concatenates in the same order.
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            l1.forward(stm_hidden.concat(ntm_hidden))
        });

    let schedule = TrainingSchedule {
        net_id: args.net_id.clone(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: args.batch_size,
            batches_per_superbatch: args.batches_per_superbatch,
            start_superbatch: 1,
            end_superbatch: args.superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: args.wdl },
        lr_scheduler: lr::StepLR {
            start: args.lr,
            gamma: 0.1,
            step: args.lr_step,
        },
        save_rate: args.save_rate,
    };

    // bullet keeps these as borrowed strings for the whole run; leaking two
    // short strings once at startup is the simplest way to give them that
    // lifetime.
    let output_directory: &'static str = Box::leak(args.out.into_boxed_str());
    let data_path: &'static str = Box::leak(args.data.into_boxed_str());

    let settings = LocalSettings {
        threads: args.threads,
        test_set: None,
        output_directory,
        batch_queue_size: 64,
    };

    let data_loader = loader::DirectSequentialDataLoader::new(&[data_path]);
    trainer.run(&schedule, &settings, &data_loader);
}
