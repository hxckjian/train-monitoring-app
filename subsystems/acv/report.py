"""Create a lightweight SVG feature plot and audit unlabelled test inputs."""
import json
from pathlib import Path
import pandas as pd
from .features import load_input
from .train import inspect_case

MODULE_DIR = Path(__file__).resolve().parent

def main():
    features = pd.read_csv(MODULE_DIR / 'training_features.csv', dtype={'car': str})
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="980" height="660" viewBox="0 0 980 660">',
           '<rect width="980" height="660" fill="white"/>',
           '<g font-family="Arial" fill="#172b4d"><text x="30" y="30" font-size="21">Cooling cabin excess over same-time peer median</text>',
           '<text x="30" y="53" font-size="14">Median degrees Celsius by car. Orange: labelled fault. Grey dash: missing.</text>']
    for i, (name, frame) in enumerate(features.groupby('file_id', sort=True)):
        left, top = 35 + (i % 2) * 480, 95 + (i // 2) * 185
        svg.append(f'<text x="{left}" y="{top}" font-size="16">{name}</text>')
        zero = top + 105
        svg.append(f'<line x1="{left}" y1="{zero}" x2="{left+405}" y2="{zero}" stroke="#778899"/>')
        for j, row in enumerate(frame.itertuples()):
            x = left + 12 + 50*j
            v = row.peer_median
            color = '#ce5d19' if row.faulty else '#42789e'
            svg.append(f'<text x="{x}" y="{top+146}" font-size="13">{row.car}</text>')
            if pd.isna(v):
                svg.append(f'<text x="{x}" y="{zero-5}" fill="#777">—</text>')
            else:
                height = abs(v)*40
                y = zero-height if v>=0 else zero
                svg.append(f'<rect x="{x}" y="{y:.2f}" width="28" height="{max(height,1):.2f}" fill="{color}"/>')
                svg.append(f'<text x="{x}" y="{y-6:.2f}" font-size="11">{v:.2f}</text>')
    svg.append('</g></svg>')
    (MODULE_DIR / 'eda_features.svg').write_text('\n'.join(svg), encoding='utf-8')
    # Normalize earlier reports too; JSON null denotes absent data.
    eda_path = MODULE_DIR / 'eda.json'
    eda = json.loads(eda_path.read_text(), parse_constant=lambda _: None)
    eda_path.write_text(json.dumps(eda, indent=2, allow_nan=False), encoding='utf-8')
    test_dir = MODULE_DIR.parents[1] / '02_Datasets' / 'ACV' / 'Test'
    test = [inspect_case(load_input(path), path) for path in sorted(test_dir.glob('*.xlsx'))]
    (MODULE_DIR / 'test_eda.json').write_text(json.dumps(test, indent=2), encoding='utf-8')
    print('Saved EDA plot and test audit. No fitting or model changes.')

if __name__ == '__main__':
    main()
