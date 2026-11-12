"""Replayable adaptive Repair trigger; never changes quality gates or BO acquisition."""
from agents.tuning.dqbo.src.services.engine.pareto import hypervolume_3d

LEVELS = (0.0, 0.05, 0.10, 0.20)


def threshold_state(history, parent_trial_id):
    """Replay prior native rounds, counting complete measured episodes once.

    Technical/unmeasured and budget-truncated episodes are neutral. State is
    derived from authoritative history so retries/restarts cannot double count.
    Five non-triggering native rounds lower one level and clear partial strikes.
    """
    level = strikes = idle = 0
    transitions = []
    episodes = {e['parent_trial_id']: e for e in history.get('repair_episodes', [])}
    for trial in history.get('trials', []):
        if trial['trial_id'] == parent_trial_id:
            break
        if trial.get('trial_role') != 'native_bo':
            continue
        before = level
        episode = episodes.get(trial['trial_id'])
        reason = None
        if episode is None:
            idle += 1
            if idle >= 5:
                level = max(0, level - 1)
                strikes = idle = 0
                reason = 'five_native_rounds_without_repair'
        else:
            idle = 0
            rows = [r for r in history.get('trials', [])
                    if r.get('trial_role') == 'repair'
                    and r.get('parent_trial_id') == trial['trial_id']]
            if episode.get('successful_trial_ids') or any(r.get('repair_success') is True for r in rows):
                level = max(0, level - 1)
                strikes = 0
                reason = 'measured_quality_and_pareto_success'
            elif episode.get('status') == 'failed_after_round_2' and any(r.get('status') == 'complete' for r in rows):
                strikes += 1
                if strikes == 3:
                    level = min(len(LEVELS) - 1, level + 1)
                    strikes = 0
                    reason = 'three_completed_episodes_without_gain'
        if reason:
            transitions.append({'after_parent': trial['trial_id'], 'reason': reason,
                                'from_level': before, 'to_level': level})
    return {'level': level, 'relative_threshold': LEVELS[level],
            'failures_since_last_level_change': strikes,
            'native_rounds_without_repair': idle, 'transitions': transitions}


def assess_trigger(history, parent_trial_id, front, reference, gain, epsilon):
    state = threshold_state(history, parent_trial_id)
    volume = hypervolume_3d(front, reference)
    ratio = gain / volume if volume > epsilon else None
    return {**state, 'version': 'adaptive-repair-threshold.v1',
            'front_hypervolume': volume, 'relative_hv_gain': ratio,
            'reference_point': list(reference),
            'fallback_to_absolute_gate': ratio is None,
            'passed': ratio is None or ratio > state['relative_threshold']}
