"""Final merge/delete stage; raw hypotheses remain in the source audit bundle."""
from copy import deepcopy

from .cluster_experience import _hash, statistics, front_and_good

VERSION = 'consistent-high-performance-merged-triplets-top4.v1'


def select_observations(bundle):
    rows = {r['trial_id']: r for r in bundle['evidence']}
    minimum = bundle['policy']['minimum_unique_support']
    selected, deleted, groups = [], [], {}
    for report in bundle['experiences']:
        kind = report['type']
        if report['disposition'] == 'discard':
            deleted.append(dict(source_id=report['experience_id'], reason='model_discard'))
        elif kind == 'high_performance_pattern':
            consistent = (report['evidence_status'] == 'observed_pattern'
                and report['disposition'] == 'keep'
                and report['unique_support_configurations'] >= minimum
                and not any(report[k] for k in ('counterexample_trial_ids',
                    'boundary_counterexample_trial_ids', 'unknown_trial_ids', 'runtime_failure_trial_ids')))
            if consistent:
                selected.append(deepcopy(report))
            else:
                deleted.append(dict(source_id=report['experience_id'], reason='not_consistent_multi_configuration'))
        elif kind == 'quality_triplet_preference':
            groups.setdefault(_hash(report['triplet']), []).append(report)
        else:
            selected.append(deepcopy(report))
    merged = []
    for key, sources in sorted(groups.items()):
        def union(field):
            return sorted({t for r in sources for t in r[field]})
        matched_ids = union('matched_trial_ids')
        matched = [rows[t] for t in matched_ids]
        support = [r for r in matched if r['quality_passed'] is True]
        counter = [r for r in matched if r['quality_passed'] is False]
        boundary = [t for t in union('boundary_counterexample_trial_ids') if t not in matched_ids]
        unknown = [r for r in matched if r['quality_passed'] is None]
        perf = [r for r in support if r['performance']]
        if not perf:
            deleted.extend(dict(source_id=r['experience_id'], reason='no_feasible_performance') for r in sources)
            continue
        best = min(perf, key=lambda r: (-r['performance'][0], r['performance'][1], r['performance'][2], r['trial_id']))
        count = len({r['configuration_id'] for r in support})
        state = ('instance_only' if count < minimum else 'mixed_evidence' if counter or boundary or unknown
                 else 'observed_pattern')
        action = 'keep' if any(r['disposition'] == 'keep' for r in sources) else 'downgrade'
        if state == 'observed_pattern' and action == 'downgrade':
            state = 'tentative_pattern'
        front, _ = front_and_good(perf, 0)
        report = dict(type='quality_triplet_preference', triplet=sources[0]['triplet'],
            experience_id='merged_' + _hash([VERSION, sorted(r['experience_id'] for r in sources)])[:24],
            source_experience_ids=sorted(r['experience_id'] for r in sources),
            region_ids=sorted({r['cluster_id'] for r in sources}),
            region_semantics='union_of_original_regions_not_unconditional_triplet_rule',
            evidence_status=state, disposition=action, unique_support_configurations=count,
            support_trial_ids=[r['trial_id'] for r in support],
            matched_trial_ids=matched_ids, counterexample_trial_ids=[r['trial_id'] for r in counter],
            boundary_counterexample_trial_ids=boundary,
            runtime_failure_trial_ids=[r['trial_id'] for r in matched if r['status']=='runtime_failed'],
            unknown_trial_ids=[r['trial_id'] for r in unknown], statistics=statistics(matched),
            boundary_statistics=statistics([rows[t] for t in boundary]),
            local_high_performance_support_ids=union('local_high_performance_support_ids'),
            representative_configurations=[dict(trial_id=r['trial_id'], candidate=r['candidate'], performance=r['performance']) for r in front],
            ranking_evidence=dict(best_feasible_trial_id=best['trial_id'], best_feasible_performance=best['performance']),
            causal_repair_verified=False, validation='same_history_consistency_only',
            pass_fraction_is_future_probability=False)
        merged.append(report)
    # Rank observed achievable performance, never failed-trial throughput or
    # summed regional support (the latter would reward overlapping regions).
    merged.sort(key=lambda r: (-r['ranking_evidence']['best_feasible_performance'][0],
        r['ranking_evidence']['best_feasible_performance'][1],
        r['ranking_evidence']['best_feasible_performance'][2], _hash(r['triplet'])))
    ranking = []
    for rank, report in enumerate(merged, 1):
        report['rank'] = rank
        ranking.append(dict(rank=rank, triplet=report['triplet'], **report['ranking_evidence'],
                            retained=rank <= 4, source_ids=report['source_experience_ids']))
        if rank <= 4:
            selected.append(report)
        else:
            deleted.extend(dict(source_id=i, reason='outside_triplet_top4') for i in report['source_experience_ids'])
    receipt = dict(version=VERSION, top_k=4,
        ranking_policy='best_quality_feasible_qps_desc_then_mean_latency_asc_then_p95_asc_then_triplet_hash',
        input_count=len(bundle['experiences']), output_count=len(selected),
        deleted_candidates=deleted, triplet_ranking=ranking,
        region_candidates=[deepcopy(r) for r in bundle['experiences'] if r['type']=='quality_triplet_preference'])
    return selected, receipt
