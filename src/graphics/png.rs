// SPDX-License-Identifier: GPL-3.0-or-later
//! [stream X] PNG output: the figure's SVG rasterized with resvg (pure Rust), text in the
//! system's Helvetica/Arial. The resolution is 150 dpi (the SVG is in points), close to the
//! size MATLAB's `print -dpng` writes.

use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};

/// Pixels per point (150 dpi).
pub const SCALE: f32 = 150.0 / 72.0;

fn fontdb() -> Arc<resvg::usvg::fontdb::Database> {
    static DB: OnceLock<Arc<resvg::usvg::fontdb::Database>> = OnceLock::new();
    DB.get_or_init(|| {
        let mut db = resvg::usvg::fontdb::Database::new();
        db.load_system_fonts();
        db.set_sans_serif_family("Helvetica");
        Arc::new(db)
    })
    .clone()
}

/// Rasterizes an SVG document to PNG bytes at [`SCALE`].
pub fn svg_to_png(svg: &str) -> Result<Vec<u8>> {
    let opts = resvg::usvg::Options { fontdb: fontdb(), ..Default::default() };
    let tree = resvg::usvg::Tree::from_str(svg, &opts).map_err(|e| anyhow!("png: invalid SVG: {e}"))?;
    let size = tree.size();
    let (w, h) = ((size.width() * SCALE).ceil() as u32, (size.height() * SCALE).ceil() as u32);
    let mut pixmap = resvg::tiny_skia::Pixmap::new(w.max(1), h.max(1)).ok_or_else(|| anyhow!("png: empty figure"))?;
    pixmap.fill(resvg::tiny_skia::Color::WHITE);
    resvg::render(&tree, resvg::tiny_skia::Transform::from_scale(SCALE, SCALE), &mut pixmap.as_mut());
    pixmap.encode_png().map_err(|e| anyhow!("png: {e}"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn rasterizes_a_figure() {
        let fig = crate::graphics::Figure {
            name: "t".into(),
            title: "Test".into(),
            rows: 1,
            cols: 1,
            panels: vec![crate::graphics::Panel {
                title: "x".into(),
                series: vec![crate::graphics::Series::Line {
                    x: vec![1.0, 2.0, 3.0],
                    y: vec![0.0, 1.0, 0.5],
                    color: crate::graphics::Color::Black,
                    style: crate::graphics::LineStyle::Solid,
                    width: 1.0,
                    label: None,
                }],
                ..Default::default()
            }],
        };
        let png = super::svg_to_png(&crate::graphics::layout::render(&fig).to_svg()).unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        assert!(png.len() > 1000);
    }
}
