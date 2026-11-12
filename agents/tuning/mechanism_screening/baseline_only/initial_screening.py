"""Knowledge routing and mechanical companion protection for single-mechanism screening."""
from .combination_prompts import MECHANISMS


ALIASES = {
    'enable_prefix_caching': {'enable_prefix_caching', 'prefix_caching', 'vllm_automatic_prefix_caching', 'prefix'},
    'chunked_prefill': {'chunked_prefill', 'chunked'},
    'quantization': {'quantization'},
    'dtype': {'dtype'},
    'kv_cache_dtype': {'kv_cache_dtype'},
    'cacheblend_mode': {'cacheblend_mode', 'cacheblend'},
    'speculative_method': {'speculative_method', 'speculative_decoding', 'ngram_speculative_decoding', 'speculative'},
}


def mechanism_knowledge(knowledge, mechanism):
    """Use explicit ownership or legacy local ID namespaces, never prose matching."""
    selected = {}
    aliases = ALIASES[mechanism]
    for identifier, entry in knowledge.items():
        if isinstance(entry, dict):
            if entry.get('type') == 'hard_link':
                continue  # Legality and companion protection are program responsibilities.
            owners = set(entry.get('mechanisms', [])) | set(entry.get('group_ids', []))
            owners.add(entry.get('mechanism_id'))
            if owners & aliases:
                selected[identifier] = entry
                continue
        if any(identifier == f'local.docs.{alias}' or
               identifier.startswith((f'local.{alias}.', f'user.{alias}.')) for alias in aliases):
            selected[identifier] = entry
    return selected


def independent_exclusion_scope(catalog, mechanism, rejected_values):
    """Only prune rows with a legal alternative for this mechanism in the same background.

    Use the original legal catalog, including already pruned alternatives: earlier
    performance decisions cannot turn an independent choice into a hard constraint.
    For example, CacheBlend-on has no prefix-on alternative, so its forced
    prefix-off companion cannot be pruned by a prefix-only judgment.
    """
    neighbors = [key for key in MECHANISMS if key != mechanism]
    switchable_backgrounds = {
        tuple(row['mechanisms'][key] for key in neighbors)
        for row in catalog.values() if row['mechanisms'][mechanism] not in rejected_values
    }
    return {identifier for identifier, row in catalog.items()
            if tuple(row['mechanisms'][key] for key in neighbors) in switchable_backgrounds}
