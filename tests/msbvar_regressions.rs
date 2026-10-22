//! Regression tests for `src/msbvar` and `src/heterogeneity`: no in-memory fallback for the
//! estimate (`est_final_<file_tag>.out` must exist), `ms_compute_mdd` needs `simulation_<tag>.out` and reads
//! the estimate of `simulation_file_tag`, `filtered_probabilities` before `real_time_smoothed`, option
//! validation (svar equations, error band percentiles, freq, nlags), the tuned-then-fixed proposal of the
//! random-walk Metropolis sampler and the persistence of the `heterogeneity_simulate` options.

use std::path::{Path, PathBuf};

use dynare::heterogeneity::SimOptions;
use dynare::heterogeneity::commands::sim_options;
use dynare::modfile::{Command, OptValue, parse_mod_file};
use dynare::msbvar::mdd;
use dynare::msbvar::ms_commands;
use dynare::msbvar::ms_files::read_est_final;

fn refdir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/msbvar/ms_est")
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("msbvar_regressions_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::copy(refdir().join("d2.csv"), d.join("d2.csv")).unwrap();
    d
}

/// `tmp` plus Dynare's `est_final_none_out.out` (the estimate of `est_none.mod`).
fn tmp_with_estimate(name: &str) -> PathBuf {
    let d = tmp(name);
    std::fs::copy(refdir().join("est_final_none_out.out"), d.join("est_final_none_out.out")).unwrap();
    d
}

fn model_part() -> String {
    std::fs::read_to_string(refdir().join("est_none.mod"))
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with("ms_"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn run_text(d: &Path, fname: &str, text: &str) -> anyhow::Result<()> {
    let f = d.join(format!("{fname}.mod"));
    std::fs::write(&f, text).unwrap();
    let mf = parse_mod_file(&f, &[]).unwrap();
    for c in &mf.commands {
        if let Command::Other(n, _) = c {
            if ms_commands::handles(n) {
                ms_commands::run_ms_command(&mf, c, d, fname, 3)?;
            }
        }
    }
    Ok(())
}

fn err(r: anyhow::Result<()>) -> String {
    format!("{:#}", r.unwrap_err())
}

const EST: &str = "ms_estimation(datafile=d2, freq=4, initial_year=1960, final_year=1988, nlags=2, specification=none, output_file_tag=first, max_number_of_stages=0, convergence_starting_value=1e-5, convergence_ending_value=1e-7, number_of_large_perturbations=0, number_of_small_perturbations=0);\n";

/// A second .mod of the same process does not pick up the estimate of the first one.
#[test]
fn no_fallback_to_the_previous_estimate() {
    let d = tmp("nofallback");
    run_text(&d, "one", &(model_part() + EST)).unwrap();
    assert!(d.join("est_final_first.out").is_file());
    // file_tag defaults to the file name: est_final_two.out does not exist
    let e = err(run_text(&d, "two", &(model_part() + "ms_compute_probabilities(nograph);\n")));
    assert!(e.contains("Could not find free parameter file: est_final_two.out"), "{e}");
    let e = err(run_text(&d, "two", &(model_part() + "ms_irf(horizon=4, nograph);\n")));
    assert!(e.contains("Could not find free parameter file"), "{e}");
    // with the file it works
    run_text(&d, "two", &(model_part() + "ms_compute_probabilities(file_tag=first, nograph);\n")).unwrap();
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn mdd_needs_the_simulation_file_and_reads_the_simulation_tag_estimate() {
    let d = tmp_with_estimate("mdd");
    let e = err(run_text(&d, "m", &(model_part() + "ms_compute_mdd(file_tag=none_out);\n")));
    assert!(e.contains("Could not find Metropolis Hastings file: simulation_none_out.out"), "{e}");
    // simulation_file_tag=other: simulation_other.out exists, est_final_other.out does not -> the estimate of
    // the simulation tag is required (Dynare's C code runs with -ft simulation_file_tag)
    std::fs::write(d.join("simulation_other.out"), "").unwrap();
    let e = err(run_text(&d, "m", &(model_part() + "ms_compute_mdd(file_tag=none_out, simulation_file_tag=other);\n")));
    assert!(e.contains("est_final_other.out"), "{e}");
    // the file_tag estimate must exist too (set_ms_estimation_file)
    let e = err(run_text(&d, "m", &(model_part() + "ms_compute_mdd(file_tag=missing, simulation_file_tag=other);\n")));
    assert!(e.contains("est_final_missing.out"), "{e}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn filtered_takes_precedence_over_real_time_smoothed() {
    let d = tmp_with_estimate("filt");
    run_text(
        &d,
        "p",
        &(model_part()
            + "ms_compute_probabilities(file_tag=none_out, filtered_probabilities, real_time_smoothed, nograph);\n"),
    )
    .unwrap();
    assert!(d.join("filtered_none_out.out").is_file());
    assert!(!d.join("real_time_smoothed_none_out.out").is_file());
    run_text(&d, "p", &(model_part() + "ms_compute_probabilities(file_tag=none_out, real_time_smoothed, nograph);\n"))
        .unwrap();
    assert!(d.join("real_time_smoothed_none_out.out").is_file());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn svar_equations_out_of_range_fail() {
    for (eq, what) in [("0", "zero"), ("-1", "negative"), ("3", "above nvar"), ("[1 3]", "vector above nvar")] {
        let d = tmp(&format!("eq_{what}").replace(' ', "_"));
        let text = model_part()
            .replace("svar(variances, chain=1);", &format!("svar(variances, equations={eq}, chain=1);"))
            + EST;
        assert!(text.contains("equations="));
        let e = err(run_text(&d, "eq", &text));
        assert!(e.contains("equation"), "equations={eq}: {e}");
        let _ = std::fs::remove_dir_all(&d);
    }
}

#[test]
fn error_band_percentiles_must_lie_in_the_unit_interval() {
    let d = tmp_with_estimate("bands");
    for cmd in ["ms_irf", "ms_forecast"] {
        for b in ["[0 0.5]", "[0.16 1]", "[1.5]"] {
            let e = err(run_text(
                &d,
                "b",
                &(model_part()
                    + &format!("{cmd}(file_tag=none_out, horizon=2, error_band_percentiles={b}, nograph);\n")),
            ));
            assert!(e.contains("error_band_percentiles"), "{cmd} {b}: {e}");
        }
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn freq_and_nlags_must_be_positive() {
    let d = tmp("freq");
    let e = err(run_text(&d, "f", &(model_part() + &EST.replace("freq=4", "freq=0"))));
    assert!(e.contains("freq"), "{e}");
    let e = err(run_text(&d, "f", &(model_part() + &EST.replace("nlags=2", "nlags=0"))));
    assert!(e.contains("nlags"), "{e}");
    let _ = std::fs::remove_dir_all(&d);
}

/// The proposal is tuned during the `adaptive` draws, which precede the burn-in and the kept draws: with no
/// burn-in, the first kept draw is already past the calibration phase, and the kept draws are the tail of a
/// longer run with the same calibration (fixed proposal afterwards).
#[test]
fn rwmh_tunes_before_burn_in() {
    let est = read_est_final(&refdir().join("est_final_none_out.out")).unwrap();
    let (a, _) = mdd::rwmh(&est.setup, &est.theta, 50, 0, 1, 2000, 5);
    let (b, lb) = mdd::rwmh(&est.setup, &est.theta, 30, 20, 1, 2000, 5);
    assert_eq!((a.nrows(), b.nrows(), lb.len()), (50, 30, 30));
    for i in 0..30 {
        for j in 0..a.ncols() {
            assert_eq!(a[(20 + i, j)], b[(i, j)]);
        }
    }
    // the calibration draws are not returned: the first kept draw has moved away from the mode
    assert!((0..a.ncols()).any(|j| a[(0, j)] != est.theta[j]));
}

#[test]
fn heterogeneity_simulate_options_persist() {
    let first = vec![
        ("irf".to_string(), OptValue::Num(30.0)),
        ("periods".to_string(), OptValue::Num(25.0)),
        ("relative_irf".to_string(), OptValue::Flag),
        ("irf_plot_threshold".to_string(), OptValue::Num(1e-6)),
        ("irf_shocks".to_string(), OptValue::List(vec![OptValue::Str("e".into())])),
        (dynare::modfile::RAW_ARGS_KEY.to_string(), OptValue::List(vec![OptValue::Str("c".into())])),
    ];
    let o1 = sim_options(&first, &SimOptions::default()).unwrap();
    let o2 = sim_options(&vec![("periods".to_string(), OptValue::Num(10.0))], &o1).unwrap();
    assert_eq!(o2.irf, Some(30));
    assert_eq!(o2.periods, Some(10));
    assert_eq!(o2.relative_irf, Some(true));
    assert_eq!(o2.irf_plot_threshold, Some(1e-6));
    assert_eq!(o2.irf_shocks, Some(vec!["e".to_string()]));
    assert!(o2.var_list.is_empty(), "the variable list is the command's own");
}
