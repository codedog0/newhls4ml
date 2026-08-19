"""Summarize csynth reports across hls_output_<tag> directories into a table."""
import glob
import json
import sys

sys.path.insert(0, '.')
from hls4ml.report import parse_vivado_report


def main():
    dirs = sorted(glob.glob('hls_output_*')) or sorted(glob.glob('hls_output'))
    rows = []
    for d in dirs:
        tag = d.replace('hls_output_', '').replace('hls_output', 'base')
        try:
            report = parse_vivado_report(d)
        except Exception as e:
            rows.append({'tag': tag, 'error': str(e)})
            continue
        if not report or 'CSynthesisReport' not in report:
            rows.append({'tag': tag, 'error': 'no CSynthesisReport (csynth may have failed or not run)'})
            continue
        csr = report['CSynthesisReport']
        rows.append({
            'tag': tag,
            'BestLatency': csr.get('BestLatency'),
            'WorstLatency': csr.get('WorstLatency'),
            'IntervalMin': csr.get('IntervalMin'),
            'IntervalMax': csr.get('IntervalMax'),
            'EstimatedClockPeriod': csr.get('EstimatedClockPeriod'),
            'BRAM_18K': csr.get('BRAM_18K'),
            'DSP': csr.get('DSP'),
            'FF': csr.get('FF'),
            'LUT': csr.get('LUT'),
            'URAM': csr.get('URAM'),
        })

    print(json.dumps(rows, indent=2))

    # Markdown table
    cols = ['tag', 'BestLatency', 'WorstLatency', 'IntervalMin', 'IntervalMax',
            'EstimatedClockPeriod', 'LUT', 'FF', 'DSP', 'BRAM_18K']
    print('\n| ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * len(cols))
    for r in rows:
        print('| ' + ' | '.join(str(r.get(c, r.get('error', ''))) for c in cols) + ' |')


if __name__ == '__main__':
    main()
