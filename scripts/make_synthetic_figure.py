"""Generate an original synthetic GPU decomposition SVG without a plotting dependency."""

from pathlib import Path

import numpy as np

from iceemdan_cupy import ICEEMDAN, to_numpy

ROOT = Path(__file__).resolve().parents[1]


def main():
    n = np.arange(256)
    x = np.sin(2 * np.pi * 0.045 * n) + 0.35 * np.sin(2 * np.pi * 0.17 * n)
    noise = np.random.default_rng(7).normal(size=(4, len(x)))
    model = ICEEMDAN(trials=4, epsilon=0.2, batch_emd=True, graph_control=False)
    parts = to_numpy(model(x, max_imf=3, noise=noise))
    assert np.allclose(parts.sum(axis=0), x, rtol=1e-9, atol=1e-10)
    rows = [("signal", x, "#1b365d")]
    rows += [(f"component {i + 1}", part, "#007c78") for i, part in enumerate(parts[:-1])]
    rows.append(("residue", parts[-1], "#be5a2c"))
    width, height, left, right, top, row_height = 900, 135 * len(rows) + 94, 145, 36, 78, 135
    plot_width = width - left - right
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Original synthetic ICEEMDAN GPU decomposition</title>',
        '<desc id="desc">A synthetic two frequency signal and its GPU components with the final residue, using four noise trials.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="145" y="35" font-family="Arial,sans-serif" font-size="21" fill="#1b365d">Synthetic GPU decomposition</text>',
        '<text x="145" y="57" font-family="Arial,sans-serif" font-size="12" fill="#52667b">4 trials · ε = 0.2 · NumPy noise seed 7 · batch EMD</text>',
    ]
    for i, (label, values, color) in enumerate(rows):
        center = top + i * row_height + 53
        scale = 40 / max(float(np.max(np.abs(values))), 1e-12)
        points = " ".join(
            f"{left + j * plot_width / (len(values) - 1):.1f},{center - value * scale:.1f}"
            for j, value in enumerate(values)
        )
        svg += [
            f'<line x1="{left}" y1="{center}" x2="{width - right}" y2="{center}" stroke="#e3e9ee"/>',
            f'<text x="18" y="{center + 4}" font-family="Arial,sans-serif" font-size="13" fill="#34495e">{label}</text>',
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.35" stroke-linejoin="round"/>',
        ]
    bottom = top + (len(rows) - 1) * row_height + 102
    svg += [
        f'<text x="{width // 2}" y="{bottom}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#52667b">sample (0–255)</text>',
        "</svg>",
    ]
    destination = ROOT / "assets" / "synthetic_decomposition.svg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
