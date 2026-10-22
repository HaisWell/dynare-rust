//! Renders a demo figure with every series kind to PDF, EPS and SVG (in
//! `$DYNARE_GRAPHICS_DEMO_DIR` if set, else the cargo test temp dir) and checks the files.

use std::path::PathBuf;

use dynare::graphics::{Color, Figure, GraphFormat, GraphOptions, LineStyle, Panel, Series, date_ticks, save};

fn demo() -> Figure {
    let x: Vec<f64> = (0..200).map(|k| -3.0 + 6.0 * k as f64 / 199.0).collect();
    let dens = |m: f64, s: f64| -> Vec<f64> {
        x.iter().map(|v| (-(v - m).powi(2) / (2.0 * s * s)).exp() / (s * (2.0 * std::f64::consts::PI).sqrt())).collect()
    };
    let prior_post = Panel {
        title: "SE_epinf_price_markup_long_name".into(),
        series: vec![
            Series::Line {
                x: x.clone(),
                y: dens(0.0, 1.0),
                color: Color::Gray,
                style: LineStyle::Solid,
                width: 2.0,
                label: None,
            },
            Series::Line {
                x: x.clone(),
                y: dens(0.4, 0.4),
                color: Color::Black,
                style: LineStyle::Solid,
                width: 2.0,
                label: None,
            },
            Series::VLine { x: 0.35, color: Color::Green, style: LineStyle::Dashed },
        ],
        ..Default::default()
    };
    let n = 94;
    let dates: Vec<String> = (0..n).map(|k| format!("{}-{:02}-28", 2003 + k / 4, 3 * (k % 4) + 3)).collect();
    let t: Vec<f64> = (1..=n).map(|k| k as f64).collect();
    let y: Vec<f64> = t.iter().map(|v| (v / 7.0).sin() * 1e-4).collect();
    let band = Panel {
        title: "y".into(),
        series: vec![
            Series::Band {
                x: t.clone(),
                lower: y.iter().map(|v| v - 3e-5).collect(),
                upper: y.iter().map(|v| v + 3e-5).collect(),
                color: Color::Gray,
                alpha: 0.4,
                label: Some("90% HPD".into()),
            },
            Series::Line {
                x: t.clone(),
                y: y.clone(),
                color: Color::Black,
                style: LineStyle::Solid,
                width: 1.0,
                label: Some("mean".into()),
            },
            Series::HLine { y: 0.0, color: Color::Red, style: LineStyle::Solid },
        ],
        x_ticks: Some(date_ticks(&dates)),
        legend: true,
        ..Default::default()
    };
    let bars = Panel {
        title: "shock decomposition".into(),
        series: vec![Series::StackedBars {
            x: t[..20].to_vec(),
            components: (0..3)
                .map(|c| (0..20).map(|k| ((k + c) as f64 / 3.0).cos() * (c as f64 + 1.0)).collect())
                .collect(),
            labels: vec!["a".into(), "b".into(), "c".into()],
        }],
        legend: true,
        ..Default::default()
    };
    let mut yy: Vec<f64> = x.iter().map(|v| -1500.0 - 40.0 * v * v).collect();
    yy[10] = f64::NAN;
    let check = Panel {
        title: "rho".into(),
        series: vec![
            Series::Line {
                x: x.clone(),
                y: yy,
                color: Color::Palette(0),
                style: LineStyle::Solid,
                width: 1.0,
                label: Some("log-post".into()),
            },
            Series::VLine { x: 0.0, color: Color::Rgb(0, 190, 190), style: LineStyle::Solid },
            Series::Points { x: vec![x[10]], y: vec![-1860.0], color: Color::Red, label: None },
        ],
        x_label: Some("value".into()),
        y_label: Some("kernel".into()),
        ..Default::default()
    };
    Figure {
        name: "Demo1".into(),
        title: "dynare-rust graphics demo".into(),
        rows: 2,
        cols: 2,
        panels: vec![prior_post, band, bars, check],
    }
}

#[test]
fn renders_all_formats() {
    let dir = std::env::var_os("DYNARE_GRAPHICS_DEMO_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("graphics_demo"));
    let opts = GraphOptions { enabled: true, formats: vec![GraphFormat::Pdf, GraphFormat::Eps, GraphFormat::Svg], dir };
    let files = save(&demo(), "demo", &opts).unwrap();
    assert_eq!(files.len(), 3);
    for f in &files {
        let b = std::fs::read(f).unwrap();
        assert!(b.len() > 1000, "{}", f.display());
    }
    assert!(files[0].ends_with("demo_Demo1.pdf"));
    let off = GraphOptions { enabled: false, ..opts };
    assert!(save(&demo(), "demo", &off).unwrap().is_empty());
}
