"""Rank frozen observations by their recorded configuration regions, without filtering."""
from copy import deepcopy


def rank_memory(memory, candidate, space):
    result = deepcopy(memory)
    target = space.bo_to_canonical(candidate)
    if 'tensor_parallel_size' in target:
        target['tp'] = target['tensor_parallel_size']
    observations = result.get('observations', [])
    success_positions = [i for i, row in enumerate(observations)
                         if row.get('observation', {}).get('category') == 'quality_triplet_preference']
    successes = [observations[i] for i in success_positions]
    for row in observations:
        if row.get('observation', {}).get('category') != 'quality_triplet_preference':
            row.pop('applicability_ranking', None)
    for row in successes:
        regions = row.get('observation', {}).get('applicability', {}).get('configuration_regions') or {}
        matches = []
        for region in regions.get('clusters', []):
            partition = region.get('partition', {})
            values = region.get('observed_values', {})
            details = {}
            skipped = []
            for key, allowed in {**values, **{k: [v] for k, v in partition.items()}}.items():
                value = target.get(key)
                if value is None or not allowed:
                    skipped.append(key)
                    continue
                elif value in allowed:
                    details[key] = 1.0
                elif isinstance(value, (int, float)) and not isinstance(value, bool) and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in allowed
                ):
                    domain = regions.get('space', {}).get(key, {})
                    width = domain.get('max', max(allowed)) - domain.get('min', min(allowed))
                    details[key] = max(0.0, 1.0 - min(abs(value-v) for v in allowed) / width) if width > 0 else 0.0
                else:
                    details[key] = 0.0
            compared_partition = [k for k in partition if k in details]
            partition_score = sum(details[k] for k in compared_partition) / len(compared_partition) if compared_partition else -1.0
            score = sum(details.values()) / len(details) if details else -1.0
            matches.append({'region_id': region.get('cluster_id'), 'partition_match': partition_score,
                            'configuration_similarity': score, 'field_scores': details,
                            'compared_field_count': len(details), 'skipped_fields': skipped,
                            'status': 'comparable' if details else 'unknown'})
        best = max(matches, key=lambda m: m['configuration_similarity'], default={})
        row['applicability_ranking'] = {'best_region': best, 'policy': 'equal_weight_all_comparable_fields; missing_is_skipped; soft_evidence_only'}
    successes.sort(key=lambda r: (
        -r['applicability_ranking']['best_region'].get('configuration_similarity', -1),
        -float(r.get('similarity') or 0),
        r.get('observation', {}).get('memory_id', '')))
    for i, row in enumerate(successes, 1):
        row['applicability_ranking']['rank'] = i
    for position, row in zip(success_positions, successes):
        observations[position] = row
    return result
