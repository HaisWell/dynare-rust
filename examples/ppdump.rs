// SPDX-License-Identifier: GPL-3.0-or-later
// stream PP debugging helper: prints the transformed model of a .mod file.
fn main() {
    let p = std::env::args().nth(1).expect("usage: ppdump file.mod");
    let mf = dynare::modfile::parse_mod_file(std::path::Path::new(&p), &[]).unwrap();
    let m = &mf.model;
    println!("endo: {:?}", m.endo_names);
    for i in 0..m.n_equations() {
        println!("{:3}: {}   {:?}", i + 1, m.equation_string(i), m.equation_tags[i]);
    }
    for a in &m.aux_vars {
        println!(
            "aux {} ty {} {:?} {:?} {}",
            m.endo_names[a.endo_index], a.ty, a.orig_index, a.orig_lead_lag, a.orig_expr
        );
    }
    println!("lli {:?}", m.lead_lag_incidence);
}
