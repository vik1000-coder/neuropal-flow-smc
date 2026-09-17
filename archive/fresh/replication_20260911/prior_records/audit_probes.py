"""Independent audit probes. Does not modify production or frozen results."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from conditional_neural_benchmark.chemical_encoding_runner import validate_resume_manifest
from conditional_neural_benchmark.inference import load_checkpoint
from conditional_neural_benchmark.data import FoldScaler, load_cohort
from conditional_neural_benchmark.runner import _split_indices
from compatibility_neural_benchmark.core import RepairedResponseConfig
from compatibility_neural_benchmark.progressive_smc import progressive_smc_repaired_responses

OUT = Path(__file__).resolve().parent
ATLAS = ROOT / 'results/neural_prediction_atlas_20260829'
torch.set_num_threads(1)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_ledger(root):
    rows = []
    for line in (root / 'checksums.sha256').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = root / name.lstrip('*')
        rows.append({'path': str(path.relative_to(ROOT)), 'ok': path.exists() and digest(path) == expected})
    return {'checked': len(rows), 'failed': [row for row in rows if not row['ok']]}


def checkpoint_probe():
    models = json.loads((ATLAS / 'canonical/models.json').read_text())
    print('models keys', list(models), flush=True)
    checkpoints = models['checkpoints']
    for row in checkpoints:
        print('checkpoint entry keys', list(row), flush=True)
        break
    path = Path(checkpoints[0]['path'])
    model, checkpoint, _ = load_checkpoint(path, 'cpu')
    torch.manual_seed(100)
    lag, d = int(checkpoint['lag']), len(checkpoint['neurons'])
    channels = int(checkpoint['input_channels'])
    x = torch.randn(1, lag * channels, requires_grad=True)
    context = model.context(x)
    direction = torch.randn_like(context)
    gradient = torch.autograd.grad((context * direction).sum(), x)[0].reshape(lag, channels)
    original = context.detach()
    changed = x.detach().reshape(1, lag, channels).clone()
    changed[:, :lag-31, :d] += 1
    delta = float((model.context(changed.reshape(1, -1)) - original).detach().abs().max())
    return {
        'path': str(path), 'checkpoint_hash_matches': digest(path) == checkpoints[0]['sha256'],
        'model_config': checkpoint['model_config'],
        'lag': lag,
        'pre_last_31_gradient_l1': float(gradient[:lag-31].abs().sum()),
        'last_31_gradient_l1': float(gradient[lag-31:].abs().sum()),
        'context_max_change_when_only_older_frames_change': delta,
        'checkpoint_keys': list(checkpoint),
    }


def resume_probe():
    source = ROOT / 'results/conditional_distribution_benchmark/chemical_encoding_primary_20260828_oh16230/manifest.json'
    manifest = json.loads(source.read_text())
    old_lag = manifest['lag_frames']
    manifest['lag_frames'] = old_lag + 1
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'manifest.json'
        path.write_text(json.dumps(manifest))
        accepted = validate_resume_manifest(path, manifest['stimulus_schema_fingerprint'], manifest['fold_assignments_sha256'])
    return {'actual_run_lag': old_lag, 'altered_lag_accepted_by_guard': accepted['lag_frames'],
            'manifest_keys': list(manifest), 'source_manifest_sha256': digest(source)}


class LinearAdapter:
    lag = 1
    device = torch.device('cpu')
    A = np.array([[.70, .10], [.35, .45]], dtype=np.float64)
    sigma = .45

    @classmethod
    def sample_standardized_next(cls, neural_history, stimulus_history, *, seed):
        generator = torch.Generator(device='cpu').manual_seed(seed)
        noise = torch.randn((len(neural_history), 2), generator=generator)
        a = torch.as_tensor(cls.A, dtype=torch.float32)
        return neural_history[:, -1] @ a.T + cls.sigma * noise


def exact_response(config, projection, low, high, bandwidth):
    """Gaussian path precision plus quadratic anchor and averaged-source clamp."""
    d, r = 2, config.repair_frames
    transition = np.eye(d * r)
    for t in range(1, r):
        transition[t*d:(t+1)*d, (t-1)*d:t*d] = -LinearAdapter.A
    prior_precision = transition.T @ transition / LinearAdapter.sigma**2
    source_end = r - config.source_lag_frames
    source_start = source_end - config.source_window_frames
    responses, gaps = [], []
    for source in range(d):
        precision = prior_precision.copy()
        for t in range(r):
            mask = np.eye(d)
            if source_start <= t < source_end:
                mask[source, source] = 0
            anchor = mask @ projection @ projection.T @ mask
            precision[t*d:(t+1)*d, t*d:(t+1)*d] += config.anchor_lambda * anchor / (projection.shape[1] * r)
        c = np.zeros(d*r)
        c[np.arange(source_start, source_end)*d + source] = 1/config.source_window_frames
        precision += np.outer(c, c)/bandwidth**2
        mean_diff = np.linalg.solve(precision, c*(high-low)/bandwidth**2)
        gaps.append(c @ mean_diff)
        responses.append([np.linalg.matrix_power(LinearAdapter.A, h) @ mean_diff[-d:] for h in config.horizon_frames])
    return np.asarray(responses), np.asarray(gaps)


def gaussian_probe():
    rows = []
    projection = np.array([[1, .25], [.15, 1.2]], dtype=np.float32)
    for lag in (1, 4, 16):
        for particles in (32, 128, 1024):
            config = RepairedResponseConfig(history_frames=1, repair_frames=lag+4,
                source_window_frames=4, source_lag_frames=lag, horizon_frames=(1, 4),
                n_particles=particles, anchor_lambda=.25, epsilon_iqr_fraction=.25,
                min_ess=6, sampling_chunk_size=1024)
            exact, gap = exact_response(config, projection, -.6, .6, .25)
            estimates, gap_estimates, validity, genealogy = [], [], [], []
            for repeat in range(16):
                result = progressive_smc_repaired_responses(LinearAdapter(),
                    np.zeros((lag+12, 2), dtype=np.float32), np.zeros(lag+12, dtype=np.float32),
                    cut_time=lag+6, projection=projection,
                    source_low=np.full(2, -.6), source_high=np.full(2, .6),
                    source_iqr=np.ones(2), thresholds=np.zeros(2), config=config,
                    seed=50003 + 8191*repeat, branch_factor=2, future_branch_factor=1)
                estimates.append(result['response_endpoint_mean'])
                gap_estimates.append(result['diagnostic_achieved_gap'])
                validity.append(result['diagnostic_valid'])
                genealogy.append(np.minimum(result['diagnostic_distinct_ancestors_low'], result['diagnostic_distinct_ancestors_high'])/particles)
            array = np.asarray(estimates)
            row = {'lag': lag, 'particles': particles, 'repeats': len(array),
                'oracle': exact.tolist(), 'estimate_mean': array.mean(axis=0).tolist(),
                'mean_bias_rmse': float(np.sqrt(np.mean((array.mean(axis=0)-exact)**2))),
                'run_rmse': float(np.sqrt(np.mean((array-exact)**2))),
                'mc_se_max': float((array.std(axis=0, ddof=1)/np.sqrt(len(array))).max()),
                'oracle_source_gap': gap.tolist(), 'estimated_source_gap_mean': np.mean(gap_estimates, axis=0).tolist(),
                'valid_fraction': float(np.mean(validity)), 'ancestor_fraction_mean': float(np.mean(genealogy))}
            print('Gaussian', lag, particles, row['mean_bias_rmse'], flush=True)
            rows.append(row)
    return rows


def training_audit():
    run = ROOT / 'results/conditional_distribution_benchmark/chemical_encoding_primary_20260828_oh16230'
    trials = pd.read_csv(run/'trial_metrics.csv')
    metric = 'energy__worm_chemical_balanced'
    scores = []
    for encoding, frame in trials.groupby('stimulus_encoding'):
        scores.append({'encoding': encoding, 'saved_fold_equal': float(frame[metric].mean()),
            'equal_worm_chemical': float(np.average(frame[metric], weights=frame.n_worm_chemical_cells))})
    candidate = trials.loc[trials.stimulus_encoding == 'chemical_onehot'].set_index(['fold','seed'])[metric]
    control = trials.loc[trials.stimulus_encoding == 'chemical_onehot_subject_shuffle'].set_index(['fold','seed'])[metric]
    delta = candidate - control
    rng = np.random.default_rng(33)
    naive = delta.to_numpy()
    fold = delta.groupby('fold').mean().to_numpy()
    naive_boot = naive[rng.integers(len(naive), size=(20000, len(naive)))].mean(axis=1)
    fold_boot = fold[rng.integers(len(fold), size=(20000, len(fold)))].mean(axis=1)
    models = json.loads((ATLAS/'canonical/models.json').read_text())
    cohort = load_cohort(cohort_mode='oh16230_head')
    assignments = pd.read_csv(ROOT/'results/conditional_distribution_benchmark/atlas_blind_world_model_20260827/fold_assignments.csv')
    lookup = assignments.set_index('worm_id').outer_fold.to_dict()
    folds = np.array([lookup[worm] for worm in cohort.worm_ids])
    check_rows = []
    for entry in models['checkpoints']:
        checkpoint = torch.load(entry['path'], map_location='cpu', weights_only=False)
        train, validation, test = _split_indices(folds, checkpoint['fold'])
        scaler = FoldScaler.fit(cohort.traces[i] for i in train)
        check_rows.append({'fold': checkpoint['fold'], 'seed': checkpoint['seed'],
            'sha256_matches': digest(entry['path']) == entry['sha256'],
            'scaler_mean_max_difference': float(np.max(np.abs(scaler.mean - checkpoint['scaler']['mean']))),
            'scaler_scale_max_difference': float(np.max(np.abs(scaler.scale - checkpoint['scaler']['scale']))),
            'train_worms': len(train), 'validation_worms': len(validation), 'test_worms': len(test),
            'splits_disjoint': not (set(train)&set(validation) or set(train)&set(test) or set(test)&set(validation))})
    return {'scores': scores, 'checkpoint_checks': check_rows,
        'bootstrap_example': {'candidate':'chemical_onehot','control':'chemical_onehot_subject_shuffle',
            'seed_delta_correlation_within_folds': float(delta.unstack().corr().iloc[0,1]),
            'iid_fold_seed_10_interval': np.quantile(naive_boot, [.025,.975]).tolist(),
            'seed_averaged_fold_5_sensitivity_interval': np.quantile(fold_boot, [.025,.975]).tolist(),
            'boundary':'Fold bootstrap is a sensitivity, not a replacement for worm-level estimates with model-refit uncertainty.'}}


def raw_reconstruction():
    queue = pd.read_csv(ATLAS/'canonical/hypothesis_queue.csv')
    cases = pd.concat([queue.iloc[:1], queue.loc[(queue.channel=='endpoint_mean') & (queue.context=='baseline')].iloc[:1]])
    results = []
    with np.load(ATLAS/'canonical/worm_matrices.npz') as saved:
        worm_order = saved['worm_ids'].astype(str).tolist()
        lags = saved['source_lag_frames'].tolist()
        horizons = saved['horizon_frames'].tolist()
        for row in cases.itertuples():
            values = {}
            for run in ['progressive_n32_s1701','progressive_n32_s2903']:
                for path in (ATLAS/run/'responses').rglob(f'*ell{row.source_lag_frames}__*.npz'):
                    with np.load(path) as z:
                        raw = z[f'response_{row.channel}'][:,:,:,row.source_index,horizons.index(row.horizon_frames),row.target_index].astype(np.float32)
                        denominator = np.maximum(np.abs(z['diagnostic_achieved_gap'][:,:,:,row.source_index]),.10)
                        normalized = raw/denominator
                        by_worm = normalized.mean(axis=(1,2)) if row.context=='state_average' else normalized[:,0].mean(axis=1)
                        for worm,value in zip(z['worm_ids'].astype(str), by_worm):
                            values.setdefault(worm,[]).append(value)
            assert len(values)==17 and all(len(v)==2 for v in values.values())
            rebuilt = np.array([np.mean(values[worm],dtype=np.float32) for worm in worm_order])
            reference = saved[f'normalized__{row.channel}__{row.context}']
            print('worm array shape', reference.shape, flush=True)
            expected = reference[lags.index(row.source_lag_frames),:,horizons.index(row.horizon_frames),row.target_index,row.source_index]
            results.append({'queue_rank':int(row.queue_rank), 'channel':row.channel, 'context':row.context,
                'source':row.source_neuron,'target':row.target_neuron,
                'raw_rebuilt_mean':float(rebuilt.mean(dtype=np.float64)), 'queue_mean':float(row.mean_normalized),
                'max_worm_matrix_difference':float(np.max(np.abs(rebuilt-expected))),
                'worm_count':len(values),'seeds_per_worm':2})
    return results


def main():
    result = {'resume': resume_probe(), 'checkpoint': checkpoint_probe()}
    (OUT / 'probe_results.json').write_text(json.dumps(result, indent=2)+'\n')
    result['gaussian_oracle'] = gaussian_probe()
    result['training_audit'] = training_audit()
    result['raw_reconstruction'] = raw_reconstruction()
    result['ledgers'] = {str(root.relative_to(ROOT)): check_ledger(root) for root in (
        ATLAS, ATLAS/'canonical', ATLAS/'complete_family_evidence_20260830',
        ATLAS/'sampling_null_controls_combined8_n128_20260830/analysis')}
    predictions = pd.read_parquet(ATLAS/'canonical/prediction_cells.parquet')
    support = pd.read_parquet(ATLAS/'canonical/support_cells.parquet')
    result['artifact_counts'] = {'prediction_rows': len(predictions), 'support_rows': len(support),
        'tiers': predictions.support_tier.value_counts().to_dict(),
        'prediction_columns': len(predictions.columns)}
    (OUT / 'probe_results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['ledgers'], indent=2), flush=True)


if __name__ == '__main__':
    main()
