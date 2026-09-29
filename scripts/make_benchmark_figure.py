"""Generate the benchmark SVG from the versioned five-size results."""

import json
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "evidence" / "benchmark_cpu_batch_5_sizes.json"
    cases = json.loads(source.read_text(encoding="utf-8"))["cases"]
    if len(cases) != 5:
        raise ValueError("Expected five benchmark cases")
    ratios = [(int(case["I"]), float(case["cpu_over_gpu"])) for case in cases]
    if any(value <= 0 for _, value in ratios):
        raise ValueError("Benchmark ratios must be positive")

    left, plot_width, upper = 115, 620, 10 * ceil(max(value for _, value in ratios) / 10)
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 390" role="img" aria-labelledby="title desc">',
        '<title id="title">Measured CPU to GPU median time ratio</title>',
        '<desc id="desc">Five matched benchmark cases: '
        + ", ".join(f"{trials} trials, {value:.2f} times" for trials, value in ratios)
        + ".</desc>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="115" y="35" font-family="Arial,sans-serif" font-size="21" fill="#1b365d">Measured CPU / GPU median time</text>',
        '<text x="115" y="57" font-family="Arial,sans-serif" font-size="12" fill="#52667b">Same 1,000-sample synthetic signal and explicit noise W</text>',
    ]
    for tick in range(0, upper + 1, 10):
        x = left + plot_width * tick / upper
        svg += [
            f'<line x1="{x:.1f}" y1="82" x2="{x:.1f}" y2="337" stroke="#e3e9ee"/>',
            f'<text x="{x:.1f}" y="355" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#52667b">{tick}×</text>',
        ]
    for i, (trials, value) in enumerate(ratios):
        y = 101 + i * 48
        width = plot_width * value / upper
        svg += [
            f'<text x="20" y="{y + 20}" font-family="Arial,sans-serif" font-size="14" fill="#34495e">{trials} trials</text>',
            f'<rect x="{left}" y="{y}" width="{width:.1f}" height="28" rx="4" fill="#007c78"/>',
            f'<text x="{left + width + 12:.1f}" y="{y + 20}" font-family="Arial,sans-serif" font-size="14" fill="#1b365d">{value:.2f}×</text>',
        ]
    svg += [
        '<text x="450" y="380" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#52667b">Serial CPU median / batch GPU median · higher means faster GPU</text>',
        "</svg>",
    ]
    destination = ROOT / "assets" / "benchmark_speedup.svg"
    destination.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
