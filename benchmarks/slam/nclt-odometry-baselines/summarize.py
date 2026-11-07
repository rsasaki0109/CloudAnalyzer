"""Build descriptive summaries/figure from all frozen runs, without selecting winners."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def summarize():
    runs = json.loads((ROOT/'results.json').read_text())['runs']
    output = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = {'cloudanalyzer':'#1565c0', 'kiss-icp':'#d84315', 'small-gicp':'#2e7d32'}
    for ax, session in zip(axes, ('april', 'june')):
        for driver, color in colors.items():
            rows = [r for r in runs if r['session'] == session and r['driver'] == driver]
            assert len(rows) == 3
            values = {
                'ate_rmse_m': [r['accuracy']['ate_m']['rmse'] for r in rows],
                'rpe_translation_rmse_m': [r['accuracy']['rpe_translation_m']['rmse'] for r in rows],
                'rpe_rotation_rmse_deg': [r['accuracy']['rpe_rotation_deg']['rmse'] for r in rows],
                'registration_seconds': [r['registration_seconds'] for r in rows],
                'pipeline_seconds': [r['pipeline_seconds'] for r in rows],
                'subprocess_seconds': [r['subprocess_seconds'] for r in rows],
                'frame_p95_ms': [r['frame_p95_ms'] for r in rows],
                'peak_process_rss_mib': [r['peak_process_rss_bytes']/2**20 for r in rows],
            }
            summary = {'session':session, 'driver':driver, 'frames':rows[0]['frames'],
                       'distance_m':rows[0]['distance_m'], 'repeats':3}
            summary['metrics'] = {key:{'median':float(np.median(v)), 'min':min(v), 'max':max(v)}
                                  for key,v in values.items()}
            output.append(summary)
            ax.scatter(values['registration_seconds'], values['ate_rmse_m'], color=color, label=driver, alpha=.7)
        ax.set_title(f'NCLT {session.title()} (existing sparse cohort)')
        ax.set_xlabel('Registration time per sequence (s, log scale)')
        ax.set_ylabel('Rigid-aligned ATE RMSE (m)')
        ax.set_xscale('log')
        ax.grid(True, alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Three fresh processes per method/session; lower-left is better. Not a SOTA ranking.', fontsize=10)
    fig.savefig(ROOT/'comparison.png', dpi=160)
    plt.close(fig)
    (ROOT/'summary.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    for row in output:
        m = row['metrics']
        print(row['session'], row['driver'], *(f'{m[k]["median"]:.6f}' for k in
              ('ate_rmse_m','rpe_translation_rmse_m','rpe_rotation_rmse_deg','registration_seconds','peak_process_rss_mib')))


if __name__ == '__main__':
    summarize()
