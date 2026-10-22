// SPDX-License-Identifier: GPL-3.0-or-later
//! Text metrics of the PDF/PostScript base-14 fonts Helvetica and Helvetica-Bold (Adobe AFM
//! widths, 1/1000 em, WinAnsi/ISO Latin-1 code points). The figures use these fonts so that
//! PDF and EPS need no embedded font program; SVG asks for Helvetica/Arial, which have the
//! same metrics.

/// Widths of Helvetica for U+0020..U+007E.
const HELVETICA: [u16; 95] = [
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, // ' '../
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, // 0-9
    278, 278, 584, 584, 584, 556, 1015, // :;<=>?@
    667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, // A-M
    722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, // N-Z
    278, 278, 278, 469, 556, 333, // [\]^_`
    556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, // a-m
    556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, // n-z
    334, 260, 334, 584, // {|}~
];

/// Widths of Helvetica-Bold for U+0020..U+007E.
const HELVETICA_BOLD: [u16; 95] = [
    278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278, // ' '../
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, // 0-9
    333, 333, 584, 584, 584, 611, 975, // :;<=>?@
    722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, // A-M
    722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, // N-Z
    333, 278, 333, 584, 556, 333, // [\]^_`
    556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, // a-m
    611, 611, 611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, // n-z
    389, 280, 389, 584, // {|}~
];

/// Width of `s` in points at font size `size`.
pub fn text_width(s: &str, size: f64, bold: bool) -> f64 {
    let table = if bold { &HELVETICA_BOLD } else { &HELVETICA };
    let w: u32 = s
        .chars()
        .map(|c| {
            let u = c as u32;
            if (0x20..0x7f).contains(&u) {
                table[(u - 0x20) as usize] as u32
            } else if u == 0xd7 || u == 0x2212 {
                584 // multiplication and minus signs
            } else {
                556
            }
        })
        .sum();
    w as f64 * size / 1000.0
}

/// Cap height and descender of Helvetica, in em.
pub const CAP_HEIGHT: f64 = 0.718;
pub const DESCENT: f64 = 0.207;

/// The WinAnsi (PDF) / ISO Latin-1 (PostScript) byte of `c`, `?` when there is none.
pub fn latin1_byte(c: char) -> u8 {
    let u = c as u32;
    match u {
        0x20..=0x7e | 0xa0..=0xff => u as u8,
        0x2212 => b'-',
        _ => b'?',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths() {
        assert_eq!(text_width("0", 10.0, false), 5.56);
        assert!((text_width("SE_eps", 10.0, false) - (6.67 + 6.67 + 5.56 + 5.56 + 5.56 + 5.0)).abs() < 1e-9);
        assert!(text_width("W", 10.0, true) > text_width("i", 10.0, true));
    }
}
