"""Independent two-layer BO memory. SQLite transactions publish complete snapshots."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .cluster_experience import _hash, audit_bundle, normalize_workload_anchor, similar_workload_anchor_similarity

SCHEMA = 'sembench.local-bo-memory.v1'
DEFAULT_ROOT = Path(__file__).resolve().parents[5] / 'my_storage' / 'bo_experience_memory'


def _connect(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / 'memory.sqlite3', timeout=30)
    db.executescript('''
        CREATE TABLE IF NOT EXISTS snapshots (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY, snapshot TEXT NOT NULL, layer TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS active_snapshots (source TEXT PRIMARY KEY, snapshot TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS memory_snapshot ON memories(snapshot, layer);
    ''')
    return db


def structure_bundle(bundle):
    if not audit_bundle(bundle)['passed']:
        raise ValueError('refusing unaudited summary')
    from .observation_selection import select_observations, VERSION
    reports, selection = (bundle["observations"], bundle["observation_generation"]) if "observations" in bundle else select_observations(bundle)
    snapshot = _hash([bundle, VERSION])  # Include final merge/delete policy in version identity.
    source = bundle.get('source_context_path') or bundle['source_run_id']
    trial_ids = {r['trial_id']: 'exp_' + _hash([snapshot, r['trial_id']])[:24] for r in bundle['evidence']}
    experiences = [dict(schema=SCHEMA, memory_type='experience', memory_id=trial_ids[r['trial_id']],
        snapshot_id=snapshot, workload_anchor=bundle['workload_anchor'],
        source_run_id=bundle['source_run_id'], trial_id=r['trial_id'],
        quality_contract=bundle['quality_contract'], trial_evidence=r) for r in bundle['evidence']]
    observations = []
    clusters = {c['cluster_id']: c for c in bundle['clustering']['clusters']}
    for report in reports:
        links = {}
        for field in ('support_trial_ids', 'counterexample_trial_ids', 'boundary_counterexample_trial_ids',
                      'runtime_failure_trial_ids', 'unknown_trial_ids', 'matched_trial_ids'):
            links[field.replace('_trial_ids', '_experience_ids')] = [trial_ids[t] for t in report[field]]
        conditions = dict(environment=bundle.get('environment'), quality_contract=bundle['quality_contract'],
                          predicates=report.get('conditions'), triplet=report.get('triplet'))
        if report.get('cluster_id'):
            conditions['configuration_region'] = dict(cluster=clusters[report['cluster_id']],
                space=bundle['clustering']['configuration_space'],
                cluster_diameter=bundle['policy']['cluster_diameter'],
                neighbor_distance=bundle['policy']['neighbor_distance'])
        if report.get('region_ids'):
            conditions['configuration_regions'] = dict(
                operator='OR', clusters=[clusters[cid] for cid in report['region_ids']],
                space=bundle['clustering']['configuration_space'],
                cluster_diameter=bundle['policy']['cluster_diameter'],
                neighbor_distance=bundle['policy']['neighbor_distance'])
        observations.append(dict(schema=SCHEMA, memory_type='observation',
            memory_id='obs_' + _hash([snapshot, report['experience_id']])[:24], snapshot_id=snapshot,
            category=report['type'], workload_anchor=bundle['workload_anchor'],
            applicability=conditions, evidence_links=links,
            disposition=report['disposition'], evidence_status=report['evidence_status'],
            # Keep the complete audited structured claim, not just LLM prose.
            structured_claim=report))
    manifest = dict(schema=SCHEMA, snapshot_id=snapshot, source_summary_snapshot_id=bundle["snapshot_id"], source=source,
        source_run_id=bundle['source_run_id'], workload_anchor=bundle['workload_anchor'],
        quality_contract=bundle['quality_contract'], quality_reassessment=bundle.get('quality_reassessment'),
        environment=bundle.get('environment'), algorithm=bundle['algorithm'], policy=bundle['policy'],
        source_history_sha256=bundle.get('source_history_sha256'), synthesis=bundle['synthesis'],
        experience_count=len(experiences), observation_count=len(observations), final_selection=selection)
    return manifest, experiences, observations


def publish_bundle(bundle_path, root=None):
    root = Path(root) if root is not None else DEFAULT_ROOT
    bundle = json.loads(Path(bundle_path).read_text())
    manifest, experiences, observations = structure_bundle(bundle)
    manifest['source_bundle_path'] = str(Path(bundle_path).resolve())
    sid = manifest['snapshot_id']
    # Transactional source of truth; readers never see a partly published version.
    db = _connect(root)
    try:
        with db:
            previous = db.execute('SELECT payload FROM snapshots WHERE id=?', (sid,)).fetchone()
            if previous and json.loads(previous[0]) != manifest:
                raise ValueError('snapshot identity already has different content')
            for record in experiences + observations:
                payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
                old = db.execute('SELECT payload FROM memories WHERE id=?', (record['memory_id'],)).fetchone()
                if old and old[0] != payload:
                    raise ValueError('immutable memory conflict')
                db.execute('INSERT OR IGNORE INTO memories VALUES (?,?,?,?)',
                           (record['memory_id'], sid, record['memory_type'], payload))
            db.execute('INSERT OR IGNORE INTO snapshots VALUES (?,?,?)',
                       (sid, manifest['source'], json.dumps(manifest, ensure_ascii=False)))
            db.execute('INSERT OR REPLACE INTO active_snapshots VALUES (?,?)', (manifest['source'], sid))
    finally:
        db.close()
    # One self-contained structured JSON export per immutable snapshot.
    from agents.tuning.dqbo.src.services.engine.history import write_json_atomic
    export = root / 'snapshots' / sid / 'memory.json'
    write_json_atomic(export, dict(manifest=manifest, experience_memories=experiences, observation_memories=observations))
    summary = export.with_name('observations.md')
    summary.write_text(render_observations(manifest, observations), encoding='utf-8')
    return dict(root=str(root.resolve()), snapshot_id=sid, experience_count=len(experiences),
                observation_count=len(observations), structured_output_path=str(export.resolve()),
                summary_path=str(summary.resolve()))


def search_memory(workload, root=None, *, limit=4, threshold=.85, minimum_shared_features=8,
                  include_inactive=False, include_discarded=False, layer="both"):
    if layer not in {'both', 'experience', 'observation'}:
        raise ValueError('invalid memory layer')
    if limit < 1 or not 0 <= threshold <= 1 or minimum_shared_features < 1:
        raise ValueError('invalid retrieval policy')
    anchor = normalize_workload_anchor(workload)
    if not anchor.get('available'):
        raise ValueError('missing static workload features')
    root = Path(root) if root is not None else DEFAULT_ROOT
    if not (root / 'memory.sqlite3').exists():
        return []
    db = _connect(root)
    try:
        query = 'SELECT s.payload FROM snapshots s'
        if not include_inactive:
            query += ' JOIN active_snapshots a ON a.snapshot=s.id'
        matches = []
        for (payload,) in db.execute(query):
            manifest = json.loads(payload)
            other = manifest['workload_anchor']
            shared = set(anchor['similarity_features']) & set(other['similarity_features'])
            similarity = similar_workload_anchor_similarity(anchor, other)
            if len(shared) >= minimum_shared_features and similarity['score'] >= threshold:
                matches.append(dict(score=similarity['score'], manifest=manifest))
        matches.sort(key=lambda m: (-m['score'], m['manifest']['snapshot_id']))
        for match in matches[:limit]:
            rows = [json.loads(p) for (p,) in db.execute('SELECT payload FROM memories WHERE snapshot=? ORDER BY id',
                                                       (match['manifest']['snapshot_id'],))]
            match['observation_memories'] = [r for r in rows if r['memory_type']=='observation'
                                            and (include_discarded or r['disposition']!='discard')]
            refs = {i for r in match['observation_memories'] for ids in r['evidence_links'].values() for i in ids}
            match['experience_memories'] = [r for r in rows if r['memory_type']=='experience'
                                             and (layer == 'experience' or r['memory_id'] in refs)]
            if layer == 'experience':
                del match['observation_memories']
            elif layer == 'observation':
                del match['experience_memories']
        return matches[:limit]
    finally:
        db.close()


def search_observations(workload, *, categories, top_k=3, root=None):
    """Rank individual active observations, not snapshots, by workload similarity."""
    if top_k < 1:
        raise ValueError('top_k must be positive')
    matches = search_memory(workload, root=root, layer='observation', limit=100000)
    rows = []
    for match in matches:
        for observation in match['observation_memories']:
            if observation['category'] in categories:
                rows.append(dict(similarity=match['score'], observation=observation,
                                 source_run_id=match['manifest']['source_run_id']))
    rows.sort(key=lambda row: (-row['similarity'], row['observation']['memory_id']))
    return rows[:top_k]


def linked_experiences(observations, *, root=None):
    """Fetch only explicit experience links of selected immutable observations."""
    root = Path(root) if root is not None else DEFAULT_ROOT
    if not (root / 'memory.sqlite3').exists():
        return []
    links = {(o['snapshot_id'], memory_id) for o in observations
             for ids in o.get('evidence_links', {}).values() for memory_id in ids}
    db = _connect(root)
    try:
        result = []
        for snapshot, memory_id in sorted(links):
            row = db.execute("SELECT payload FROM memories WHERE id=? AND snapshot=? AND layer='experience'",
                             (memory_id, snapshot)).fetchone()
            if row:
                result.append(json.loads(row[0]))
        return result
    finally:
        db.close()


def render_observations(manifest, observations):
    lines = ['# 正式 observation', '',
        '高性能模式仅保留多配置一致且无反例者。优选参数组合跨区域合并后按最佳达标 QPS 降序取 top4；平均/P95 延迟用于同分排序。',
        '合并后区域为 OR 条件，支持 trial 去重；不把局部结论升级成无条件规则。',
        '候选合并/删除记录在 memory.json 的 manifest.final_selection 中。', '']
    for obs in observations:
        r = obs['structured_claim']
        lines += [f"## {obs['category']} / {obs['memory_id']}", '',
                  '```json', json.dumps(r, ensure_ascii=False, indent=2), '```', '']
    return '\n'.join(lines)
