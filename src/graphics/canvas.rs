// SPDX-License-Identifier: GPL-3.0-or-later
//! A small display list (paths, circles, text, clipping) and its three writers: SVG, PDF
//! (hand-written, base-14 Helvetica, deflated content stream, ExtGState alpha) and EPS
//! (PostScript level 2; transparency is flattened against the white background).
//! Coordinates are points with the origin at the top left, y pointing down.

use std::fmt::Write as _;
use std::io::Write as _;

use super::font;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rgb(pub f64, pub f64, pub f64);

impl Rgb {
    pub const BLACK: Rgb = Rgb(0.0, 0.0, 0.0);
    pub const WHITE: Rgb = Rgb(1.0, 1.0, 1.0);
    fn hex(self) -> String {
        let c = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        format!("#{:02x}{:02x}{:02x}", c(self.0), c(self.1), c(self.2))
    }
    /// `self` with opacity `alpha` over white.
    fn over_white(self, alpha: f64) -> Rgb {
        let m = |v: f64| alpha * v + (1.0 - alpha);
        Rgb(m(self.0), m(self.1), m(self.2))
    }
}

#[derive(Clone, Debug)]
pub struct Stroke {
    pub color: Rgb,
    pub width: f64,
    /// Dash pattern in points (empty = solid).
    pub dash: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Anchor {
    Start,
    Middle,
    End,
}

#[derive(Clone, Debug)]
pub enum Op {
    Path {
        pts: Vec<(f64, f64)>,
        closed: bool,
        stroke: Option<Stroke>,
        fill: Option<(Rgb, f64)>,
    },
    Circle {
        cx: f64,
        cy: f64,
        r: f64,
        fill: Rgb,
        stroke: Option<Stroke>,
    },
    /// `y` is the baseline; `rotate` is 0 or 90 (counter-clockwise, reading upwards).
    Text {
        x: f64,
        y: f64,
        s: String,
        size: f64,
        bold: bool,
        anchor: Anchor,
        rotate: bool,
        color: Rgb,
    },
    ClipPush {
        x: f64,
        y: f64,
        w: f64,
        h: f64,
    },
    ClipPop,
}

#[derive(Clone, Debug, Default)]
pub struct Canvas {
    pub width: f64,
    pub height: f64,
    pub ops: Vec<Op>,
}

impl Canvas {
    pub fn new(width: f64, height: f64) -> Canvas {
        Canvas { width, height, ops: Vec::new() }
    }

    pub fn line(&mut self, pts: Vec<(f64, f64)>, stroke: Stroke) {
        if pts.len() >= 2 {
            self.ops.push(Op::Path { pts, closed: false, stroke: Some(stroke), fill: None });
        }
    }

    pub fn polygon(&mut self, pts: Vec<(f64, f64)>, fill: Rgb, alpha: f64, stroke: Option<Stroke>) {
        if pts.len() >= 3 {
            self.ops.push(Op::Path { pts, closed: true, stroke, fill: Some((fill, alpha)) });
        }
    }

    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: Option<(Rgb, f64)>, stroke: Option<Stroke>) {
        let pts = vec![(x, y), (x + w, y), (x + w, y + h), (x, y + h)];
        self.ops.push(Op::Path { pts, closed: true, stroke, fill });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn text(&mut self, x: f64, y: f64, s: &str, size: f64, bold: bool, anchor: Anchor, rotate: bool) {
        if !s.is_empty() {
            self.ops.push(Op::Text { x, y, s: s.to_string(), size, bold, anchor, rotate, color: Rgb::BLACK });
        }
    }

    pub fn clip(&mut self, x: f64, y: f64, w: f64, h: f64) {
        self.ops.push(Op::ClipPush { x, y, w, h });
    }

    pub fn unclip(&mut self) {
        self.ops.push(Op::ClipPop);
    }

    // -----------------------------------------------------------------------
    // SVG
    // -----------------------------------------------------------------------

    pub fn to_svg(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}pt" height="{h}pt" viewBox="0 0 {w} {h}">"#,
            w = fmt(self.width),
            h = fmt(self.height)
        );
        let _ = writeln!(s, r#"<rect width="100%" height="100%" fill="white"/>"#);
        let mut clip_id = 0;
        let mut open_groups = 0;
        for op in &self.ops {
            match op {
                Op::Path { pts, closed, stroke, fill } => {
                    let mut d = String::new();
                    for (k, (x, y)) in pts.iter().enumerate() {
                        let _ = write!(d, "{}{} {}", if k == 0 { "M" } else { "L" }, fmt(*x), fmt(*y));
                    }
                    if *closed {
                        d.push('Z');
                    }
                    let fill_attr = match fill {
                        Some((c, a)) if *a < 1.0 => format!(r#"fill="{}" fill-opacity="{}""#, c.hex(), fmt(*a)),
                        Some((c, _)) => format!(r#"fill="{}""#, c.hex()),
                        None => r#"fill="none""#.to_string(),
                    };
                    let _ = writeln!(s, r#"<path d="{d}" {fill_attr}{}/>"#, svg_stroke(stroke.as_ref()));
                }
                Op::Circle { cx, cy, r, fill, stroke } => {
                    let _ = writeln!(
                        s,
                        r#"<circle cx="{}" cy="{}" r="{}" fill="{}"{}/>"#,
                        fmt(*cx),
                        fmt(*cy),
                        fmt(*r),
                        fill.hex(),
                        svg_stroke(stroke.as_ref())
                    );
                }
                Op::Text { x, y, s: t, size, bold, anchor, rotate, color } => {
                    let anchor = match anchor {
                        Anchor::Start => "start",
                        Anchor::Middle => "middle",
                        Anchor::End => "end",
                    };
                    let transform = if *rotate {
                        format!(r#" transform="rotate(-90 {} {})""#, fmt(*x), fmt(*y))
                    } else {
                        String::new()
                    };
                    let _ = writeln!(
                        s,
                        r#"<text x="{}" y="{}" font-family="Helvetica, Arial, sans-serif" font-size="{}"{} text-anchor="{anchor}" fill="{}"{transform}>{}</text>"#,
                        fmt(*x),
                        fmt(*y),
                        fmt(*size),
                        if *bold { r#" font-weight="bold""# } else { "" },
                        color.hex(),
                        xml_escape(t)
                    );
                }
                Op::ClipPush { x, y, w, h } => {
                    clip_id += 1;
                    let _ = writeln!(
                        s,
                        r#"<clipPath id="c{clip_id}"><rect x="{}" y="{}" width="{}" height="{}"/></clipPath><g clip-path="url(#c{clip_id})">"#,
                        fmt(*x),
                        fmt(*y),
                        fmt(*w),
                        fmt(*h)
                    );
                    open_groups += 1;
                }
                Op::ClipPop => {
                    if open_groups > 0 {
                        s.push_str("</g>\n");
                        open_groups -= 1;
                    }
                }
            }
        }
        for _ in 0..open_groups {
            s.push_str("</g>\n");
        }
        s.push_str("</svg>\n");
        s
    }

    // -----------------------------------------------------------------------
    // PDF / PostScript share the path syntax (bottom-left origin).
    // -----------------------------------------------------------------------

    /// Content stream operators. `ps` selects PostScript syntax; `alpha_gs` maps an
    /// alpha value to its PDF ExtGState name.
    fn page_ops(&self, ps: bool, alpha_gs: &mut Vec<f64>) -> Vec<u8> {
        let h = self.height;
        let mut out: Vec<u8> = Vec::new();
        let p = |x: f64, y: f64| format!("{} {}", fmt(x), fmt(h - y));
        let (mv, ln, cl) = if ps { ("moveto", "lineto", "closepath") } else { ("m", "l", "h") };
        let set_rgb = |c: Rgb, stroke: bool| {
            if ps {
                format!("{} {} {} setrgbcolor\n", fmt(c.0), fmt(c.1), fmt(c.2))
            } else {
                format!("{} {} {} {}\n", fmt(c.0), fmt(c.1), fmt(c.2), if stroke { "RG" } else { "rg" })
            }
        };
        let set_stroke = |st: &Stroke| {
            let mut s = set_rgb(st.color, true);
            let dash: Vec<String> = st.dash.iter().map(|d| fmt(*d)).collect();
            if ps {
                let _ = writeln!(
                    s,
                    "{} setlinewidth [{}] 0 setdash 1 setlinejoin 0 setlinecap",
                    fmt(st.width),
                    dash.join(" ")
                );
            } else {
                let _ = writeln!(s, "{} w [{}] 0 d 1 j 0 J", fmt(st.width), dash.join(" "));
            }
            s
        };
        let mut gs = |a: f64, out: &mut Vec<u8>| {
            if ps {
                return;
            }
            let k = match alpha_gs.iter().position(|v| (v - a).abs() < 1e-9) {
                Some(k) => k,
                None => {
                    alpha_gs.push(a);
                    alpha_gs.len() - 1
                }
            };
            let _ = writeln!(out, "/GS{k} gs");
        };
        for op in &self.ops {
            match op {
                Op::Path { pts, closed, stroke, fill } => {
                    let mut path = String::new();
                    for (k, (x, y)) in pts.iter().enumerate() {
                        let _ = writeln!(path, "{} {}", p(*x, *y), if k == 0 { mv } else { ln });
                    }
                    if *closed {
                        let _ = writeln!(path, "{cl}");
                    }
                    if let Some((c, a)) = fill {
                        let _ = write!(out, "{}", if ps { "gsave\n" } else { "q\n" });
                        let c = if ps && *a < 1.0 { c.over_white(*a) } else { *c };
                        if *a < 1.0 {
                            gs(*a, &mut out);
                        }
                        let _ = write!(out, "{}{}", set_rgb(c, false), path);
                        let _ = writeln!(out, "{}", if ps { "fill\ngrestore" } else { "f\nQ" });
                    }
                    if let Some(st) = stroke {
                        let _ = write!(out, "{}{}{}", set_stroke(st), path, if ps { "stroke\n" } else { "S\n" });
                    }
                }
                Op::Circle { cx, cy, r, fill, stroke } => {
                    let path = if ps {
                        format!("newpath {} {} 0 360 arc closepath\n", p(*cx, *cy), fmt(*r))
                    } else {
                        circle_pdf(*cx, h - *cy, *r)
                    };
                    let _ = write!(out, "{}{}{}", set_rgb(*fill, false), path, if ps { "fill\n" } else { "f\n" });
                    if let Some(st) = stroke {
                        let _ = write!(out, "{}{}{}", set_stroke(st), path, if ps { "stroke\n" } else { "S\n" });
                    }
                }
                Op::Text { x, y, s, size, bold, anchor, rotate, color } => {
                    let w = font::text_width(s, *size, *bold);
                    let dx = match anchor {
                        Anchor::Start => 0.0,
                        Anchor::Middle => -w / 2.0,
                        Anchor::End => -w,
                    };
                    let bytes: Vec<u8> = s.chars().map(font::latin1_byte).collect();
                    let lit = string_literal(&bytes);
                    let (fx, fy) = (*x, h - *y);
                    if ps {
                        let _ = write!(
                            out,
                            "gsave\n{}/{} {} selectfont\n{} {} translate {} rotate {} 0 moveto\n",
                            set_rgb(*color, false),
                            if *bold { "HelveticaBold-L1" } else { "Helvetica-L1" },
                            fmt(*size),
                            fmt(fx),
                            fmt(fy),
                            if *rotate { 90 } else { 0 },
                            fmt(dx)
                        );
                        out.extend_from_slice(&lit);
                        out.extend_from_slice(b" show\ngrestore\n");
                    } else {
                        let tm = if *rotate {
                            format!("0 1 -1 0 {} {} Tm", fmt(fx), fmt(fy + dx))
                        } else {
                            format!("1 0 0 1 {} {} Tm", fmt(fx + dx), fmt(fy))
                        };
                        let _ = write!(
                            out,
                            "BT\n{}/{} {} Tf\n{tm}\n",
                            set_rgb(*color, false),
                            if *bold { "F2" } else { "F1" },
                            fmt(*size)
                        );
                        out.extend_from_slice(&lit);
                        out.extend_from_slice(b" Tj\nET\n");
                    }
                }
                Op::ClipPush { x, y, w, h: hh } => {
                    if ps {
                        let _ = writeln!(
                            out,
                            "gsave newpath {} {} {} {} rectclip",
                            fmt(*x),
                            fmt(h - *y - *hh),
                            fmt(*w),
                            fmt(*hh)
                        );
                    } else {
                        let _ = writeln!(out, "q {} {} {} {} re W n", fmt(*x), fmt(h - *y - *hh), fmt(*w), fmt(*hh));
                    }
                }
                Op::ClipPop => {
                    let _ = writeln!(out, "{}", if ps { "grestore" } else { "Q" });
                }
            }
        }
        out
    }

    pub fn to_pdf(&self) -> Vec<u8> {
        let mut alphas = Vec::new();
        let content = self.page_ops(false, &mut alphas);
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(&content).expect("deflate in memory");
        let content = z.finish().expect("deflate in memory");

        let mut gs = String::new();
        for (k, a) in alphas.iter().enumerate() {
            let _ = write!(gs, "/GS{k} << /Type /ExtGState /ca {} /CA {} >> ", fmt(*a), fmt(*a));
        }
        let objects: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> /ExtGState << {gs}>> >> /Contents 4 0 R >>",
                fmt(self.width),
                fmt(self.height)
            )
            .into_bytes(),
            {
                let mut v = format!("<< /Length {} /Filter /FlateDecode >>\nstream\n", content.len()).into_bytes();
                v.extend_from_slice(&content);
                v.extend_from_slice(b"\nendstream");
                v
            },
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".to_vec(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>".to_vec(),
            b"<< /Producer (dynare-rust) >>".to_vec(),
        ];
        let mut out: Vec<u8> = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
        let mut offsets = Vec::new();
        for (k, obj) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n", k + 1).as_bytes());
            out.extend_from_slice(obj);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes());
        for o in offsets {
            out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!("trailer\n<< /Size {} /Root 1 0 R /Info 7 0 R >>\nstartxref\n{xref}\n%%EOF\n", objects.len() + 1)
                .as_bytes(),
        );
        out
    }

    pub fn to_eps(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        let _ = write!(
            out,
            "%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 {} {}\n%%HiResBoundingBox: 0 0 {} {}\n%%Creator: dynare-rust\n%%LanguageLevel: 2\n%%Pages: 1\n%%EndComments\n",
            self.width.ceil() as i64,
            self.height.ceil() as i64,
            fmt(self.width),
            fmt(self.height)
        );
        // Latin-1 versions of the two fonts
        out.extend_from_slice(
            b"/reencode { findfont dup length dict begin { 1 index /FID ne { def } { pop pop } ifelse } forall\n\
              /Encoding ISOLatin1Encoding def currentdict end definefont pop } bind def\n\
              /Helvetica-L1 /Helvetica reencode\n/HelveticaBold-L1 /Helvetica-Bold reencode\n\
              %%Page: 1 1\ngsave\n1 1 1 setrgbcolor\n",
        );
        let _ = writeln!(out, "0 0 {} {} rectfill", fmt(self.width), fmt(self.height));
        out.extend(self.page_ops(true, &mut Vec::new()));
        out.extend_from_slice(b"grestore\nshowpage\n%%Trailer\n%%EOF\n");
        out
    }
}

fn svg_stroke(st: Option<&Stroke>) -> String {
    match st {
        None => String::new(),
        Some(st) => {
            let mut s =
                format!(r#" stroke="{}" stroke-width="{}" stroke-linejoin="round""#, st.color.hex(), fmt(st.width));
            if !st.dash.is_empty() {
                let d: Vec<String> = st.dash.iter().map(|v| fmt(*v)).collect();
                let _ = write!(s, r#" stroke-dasharray="{}""#, d.join(","));
            }
            s
        }
    }
}

/// Four Bezier quarter arcs (PDF has no arc operator).
fn circle_pdf(cx: f64, cy: f64, r: f64) -> String {
    let k = 0.552_284_75 * r;
    format!(
        "{} {} m\n{} {} {} {} {} {} c\n{} {} {} {} {} {} c\n{} {} {} {} {} {} c\n{} {} {} {} {} {} c\nh\n",
        fmt(cx + r),
        fmt(cy),
        fmt(cx + r),
        fmt(cy + k),
        fmt(cx + k),
        fmt(cy + r),
        fmt(cx),
        fmt(cy + r),
        fmt(cx - k),
        fmt(cy + r),
        fmt(cx - r),
        fmt(cy + k),
        fmt(cx - r),
        fmt(cy),
        fmt(cx - r),
        fmt(cy - k),
        fmt(cx - k),
        fmt(cy - r),
        fmt(cx),
        fmt(cy - r),
        fmt(cx + k),
        fmt(cy - r),
        fmt(cx + r),
        fmt(cy - k),
        fmt(cx + r),
        fmt(cy)
    )
}

/// `(...)` string with `\`, `(`, `)` escaped and non-ASCII bytes in octal.
fn string_literal(bytes: &[u8]) -> Vec<u8> {
    let mut v = vec![b'('];
    for &b in bytes {
        match b {
            b'(' | b')' | b'\\' => {
                v.push(b'\\');
                v.push(b);
            }
            0x20..=0x7e => v.push(b),
            _ => v.extend_from_slice(format!("\\{b:03o}").as_bytes()),
        }
    }
    v.push(b')');
    v
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// Compact number for the file formats: at most 3 decimals, no trailing zeros.
pub(crate) fn fmt(v: f64) -> String {
    if !v.is_finite() {
        return "0".into();
    }
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" || s.is_empty() { "0".into() } else { s.to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Canvas {
        let mut c = Canvas::new(200.0, 100.0);
        c.rect(10.0, 10.0, 100.0, 50.0, Some((Rgb(0.5, 0.5, 0.5), 0.4)), None);
        c.line(vec![(10.0, 10.0), (110.0, 60.0)], Stroke { color: Rgb::BLACK, width: 1.0, dash: vec![3.0, 2.0] });
        c.clip(0.0, 0.0, 50.0, 50.0);
        c.ops.push(Op::Circle { cx: 20.0, cy: 20.0, r: 3.0, fill: Rgb(1.0, 0.0, 0.0), stroke: None });
        c.unclip();
        c.text(100.0, 90.0, "SE_eps (a) \u{d7}10", 10.0, true, Anchor::Middle, false);
        c.text(5.0, 50.0, "rotated", 8.0, false, Anchor::Middle, true);
        c
    }

    #[test]
    fn pdf_is_well_formed() {
        let pdf = sample().to_pdf();
        // Latin-1 view keeps byte offsets
        let s: String = pdf.iter().map(|&b| b as char).collect();
        assert!(s.starts_with("%PDF-1.4"));
        assert!(s.contains("/ExtGState << /GS0"));
        assert!(s.trim_end().ends_with("%%EOF"));
        // xref offsets point at the objects
        let xref_pos: usize = s.rsplit("startxref\n").next().unwrap().lines().next().unwrap().parse().unwrap();
        assert!(pdf[xref_pos..].starts_with(b"xref"));
        let tail = String::from_utf8_lossy(&pdf[xref_pos..]).to_string();
        for (k, line) in tail.lines().skip(3).take(7).enumerate() {
            let off: usize = line[..10].parse().unwrap();
            assert!(pdf[off..].starts_with(format!("{} 0 obj", k + 1).as_bytes()), "object {}", k + 1);
        }
    }

    #[test]
    fn eps_and_svg() {
        let eps = String::from_utf8(sample().to_eps()).unwrap();
        assert!(eps.starts_with("%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 200 100"));
        assert!(eps.contains("(SE_eps \\(a\\) \\32710) show"));
        let svg = sample().to_svg();
        assert!(svg.contains("stroke-dasharray=\"3,2\""));
        assert!(svg.contains("clip-path=\"url(#c1)\""));
        assert_eq!(svg.matches("<g ").count(), svg.matches("</g>").count());
    }
}
