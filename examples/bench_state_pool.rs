//! Benchmark: warm-state reuse vs fresh-state allocation for whisper transcription.
//!
//! Loads large-v3, then alternates N transcriptions in ONE process:
//!   1. cold: create_state() per run (old voxtype behavior)
//!   2. warm: reuse a single WhisperState across runs (new pooled behavior)
//!
//! Run: cargo run --release --example bench_state_pool --features gpu-vulkan

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

fn load_wav_16k_mono(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("open wav");
    assert_eq!(reader.spec().sample_rate, 16000, "wav must be 16kHz");
    assert_eq!(reader.spec().channels, 1, "wav must be mono");
    reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn transcribe(state: &mut whisper_rs::WhisperState, samples: &[f32]) -> String {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(4);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_no_context(true);
    if samples.len() as f32 / 16000.0 < 30.0 {
        params.set_single_segment(true);
    }
    state.full(params, samples).expect("whisper full");
    let mut text = String::new();
    for segment in state.as_iter() {
        text.push_str(segment.to_str().expect("utf8"));
    }
    text.trim().to_string()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("/home/ami/.local/share/voxtype/models/ggml-large-v3.bin");
    let wav = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("tests/fixtures/vad/speech_long.wav");
    let runs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);

    let samples = load_wav_16k_mono(wav);
    let duration = samples.len() as f32 / 16000.0;
    println!(
        "model: {} | audio: {} ({:.2}s) | runs per mode: {}",
        model, wav, duration, runs
    );

    let mut ctx_params = WhisperContextParameters::default();
    ctx_params.flash_attn(true);
    let t0 = std::time::Instant::now();
    let ctx = WhisperContext::new_with_params(model, ctx_params).expect("load model");
    println!(
        "model load (from OS cache): {:.2}s\n",
        t0.elapsed().as_secs_f32()
    );

    // -- Mode A: fresh state per transcription (old voxtype behavior) --
    let mut cold: Vec<f64> = Vec::new();
    for i in 0..runs {
        let t = std::time::Instant::now();
        let mut state = ctx.create_state().expect("create state");
        let text = transcribe(&mut state, &samples);
        let e = t.elapsed().as_secs_f64();
        drop(state);
        cold.push(e);
        eprintln!("  cold run {}: {:.3}s  {:?}", i, e, text);
    }

    // -- Mode B: one warm state reused (new pooled behavior) --
    let mut warm: Vec<f64> = Vec::new();
    let mut state = ctx.create_state().expect("create state");
    for i in 0..runs {
        let t = std::time::Instant::now();
        let text = transcribe(&mut state, &samples);
        let e = t.elapsed().as_secs_f64();
        warm.push(e);
        eprintln!("  warm run {}: {:.3}s  {:?}", i, e, text);
    }
    drop(state);

    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let cold_avg = avg(&cold);
    let warm_avg = avg(&warm);
    println!("\nresults (avg over {} runs, {:.1}s clip):", runs, duration);
    println!("  fresh-state per run : {:7.3}s", cold_avg);
    println!("  warm reused state   : {:7.3}s", warm_avg);
    println!(
        "  saved per dictation : {:7.3}s ({:.0}%)",
        cold_avg - warm_avg,
        100.0 * (cold_avg - warm_avg) / cold_avg
    );
}
