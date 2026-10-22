//! `sbvar` remainder: growth-rate files, graphs and the Gibbs-sampled log marginal data density,
//! against Dynare 7.2 on reference/msbvar/msbvar_log.mod (vlistlog=(y1)). MATLAB's results
//! (outyrqm.prn, outactqmygdata.prn, log mdd -461.74 with vlog_a0_Yao = -0.08444, 3.4258, 2.1375)
//! come from running that .mod with `dynare msbvar_log nograph`.
mod common;

use std::path::{Path, PathBuf};

use dynare::modfile::{Command, parse_mod_file};
use dynare::msbvar::{self, sbvar, sbvar_extra};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("dynare_rs_sbvar_rest_{tag}"));
    std::fs::create_dir_all(&d).unwrap();
    for f in ["msbvar_log.mod", "msbvar_data.csv"] {
        std::fs::copy(root().join("reference/msbvar").join(f), d.join(f)).unwrap();
    }
    d
}

fn numbers(s: &str) -> Vec<f64> {
    s.split_whitespace().map(|x| x.parse().unwrap()).collect()
}

fn close_files(a: &Path, b: &Path, tol: f64) {
    let (x, y) = (numbers(&std::fs::read_to_string(a).unwrap()), numbers(&std::fs::read_to_string(b).unwrap()));
    assert_eq!(x.len(), y.len(), "{}", a.display());
    for (i, (u, v)) in x.iter().zip(&y).enumerate() {
        assert!((u - v).abs() <= tol * (1.0 + v.abs()), "{} entry {i}: {u} vs {v}", a.display());
    }
}

#[test]
fn growth_rate_files_and_graphs() {
    let d = scratch("full");
    let mf = parse_mod_file(&d.join("msbvar_log.mod"), &[]).unwrap();
    let c = mf.commands.iter().find(|c| matches!(c, Command::Other(n, _) if n == "sbvar")).unwrap();
    msbvar::run_command(&mf, c, &d, "msbvar_log", 1).unwrap();
    let r = root().join("reference/msbvar");
    close_files(&d.join("outyrqm.prn"), &r.join("ref_msbvar_log_outyrqm.prn"), 1e-9);
    close_files(&d.join("outactqmygdata.prn"), &r.join("ref_msbvar_log_outactqmygdata.prn"), 1e-8);
    for f in ["sbvar_irf", "sbvar_forecast", "sbvar_growth"] {
        assert!(d.join(format!("msbvar_log/graphs/msbvar_log_{f}.pdf")).is_file(), "{f}");
    }
    // the log marginal data density is written next to the other results
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(d.join("msbvar_log/Output/sbvar_results.json")).unwrap())
            .unwrap();
    assert!(j["log_mdd"].as_f64().unwrap().is_finite());
}

#[test]
fn nograph_and_graph_format_are_honoured() {
    let d = scratch("nograph");
    let text =
        std::fs::read_to_string(d.join("msbvar_log.mod")).unwrap().replace("vlistlog=(y1))", "vlistlog=(y1), nograph)");
    std::fs::write(d.join("msbvar_log.mod"), text).unwrap();
    let mf = parse_mod_file(&d.join("msbvar_log.mod"), &[]).unwrap();
    let c = mf.commands.iter().find(|c| matches!(c, Command::Other(n, _) if n == "sbvar")).unwrap();
    msbvar::run_command(&mf, c, &d, "msbvar_log", 1).unwrap();
    assert!(!d.join("msbvar_log/graphs").exists());
    assert!(d.join("outyrqm.prn").is_file());
}

#[test]
fn gibbs_marginal_data_density_matches_dynare_to_mc_accuracy() {
    let d = scratch("mdd");
    let mf = parse_mod_file(&d.join("msbvar_log.mod"), &[]).unwrap();
    let Some(Command::Other(_, o)) = mf.commands.iter().find(|c| matches!(c, Command::Other(n, _) if n == "sbvar"))
    else {
        panic!()
    };
    let p = msbvar::prepare_sample(&mf, o, &d).unwrap();
    let res = sbvar::sbvar(&p.z, &p.settings, &p.restr).unwrap();
    let m = sbvar_extra::gibbs_mdd(&res, 7, 1000, 10000).unwrap();
    // the equations whose p(a0_k|Y, a_others) is not (or hardly) simulated
    assert!((m.vlog_a0_yao[2] - 2.1375).abs() < 1e-3, "{:?}", m.vlog_a0_yao);
    assert!((m.vlog_a0_yao[1] - 3.4258).abs() < 0.05, "{:?}", m.vlog_a0_yao);
    // the first equation is a heavy-tailed importance average (seed-to-seed std about 0.15)
    assert!((m.vlog_a0_yao[0] + 0.0844).abs() < 1.0, "{:?}", m.vlog_a0_yao);
    assert!((m.log_mdd + 461.74).abs() < 1.0, "{}", m.log_mdd);
    // deterministic pieces, at full precision (reference/msbvar/dump_msbvar_log_mdd.m)
    let r: serde_json::Value = common::json_ref(root().join("reference/msbvar/ref_msbvar_log_mdd.json"));
    let near = |a: f64, k: &str| {
        let b = r[k].as_f64().unwrap();
        assert!((a - b).abs() < 1e-8 * b.abs().max(1.0), "{k}: {a} vs {b}");
    };
    near(res.vlog_ap_ya0.unwrap(), "vlog_ap_Ya0");
    near(res.vlog_a0p.unwrap(), "vlog_a0p");
    near(res.vlog_y_a.unwrap(), "vlog_Y_a");
}
