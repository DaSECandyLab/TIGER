# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
import numpy as np
from ..design_space.design_space import DesignSpace
def parse_space_from_bayesmark(api_config) -> DesignSpace:
    """
    Parse design space of bayesmark (https://github.com/uber/bayesmark)
    """
    space  = DesignSpace()
    params = []
    for param_name in api_config:
        param_conf   = api_config[param_name]
        param_type   = param_conf['type']
        param_space  = param_conf.get('space', None)
        param_range  = param_conf.get("range", None)
        param_values = param_conf.get("values", None)

        bo_param_conf = {'name' : param_name}
        if param_type == 'int': # ignore 'log' space # TODO: support log-scale int
            bo_param_conf['type'] = 'int'
            bo_param_conf['lb']   = param_range[0]
            bo_param_conf['ub']   = param_range[1]
        elif param_type == 'bool':
            bo_param_conf['type'] = 'bool'
        elif param_type in ('cat', 'ordinal'):
            bo_param_conf['type']       = 'cat'
            bo_param_conf['categories'] = list(set(param_values))
        elif param_type == 'real':
            if param_space in ('log', 'logit'):
                bo_param_conf['type'] = 'pow'
                bo_param_conf['base'] = 10
                bo_param_conf['lb']   = param_range[0]
                bo_param_conf['ub']   = param_range[1]
            else:
                bo_param_conf['type'] = 'num'
                bo_param_conf['lb']   = param_range[0]
                bo_param_conf['ub']   = param_range[1]
        else:
            assert False, "type %s not handled in API" % param_type
        params.append(bo_param_conf)
    space.parse(params)
    return space


def ensure_hard_constr(samples, max_sequence_length):
    samples = samples.apply(prefix_chunked_update, axis=1)
    samples = samples.apply(update_max_num_batched_tokens, axis=1, max_sequence_length=max_sequence_length)
    # chunked prefill 开启时，一次 scheduler iteration 的 token budget 可以小于
    # 单条请求允许的最大上下文长度。这里不再把 max_model_len 错误压到
    # max_num_batched_tokens，否则会把合法候选改成搜索空间外的值。
    samples.loc[samples['max_num_batched_tokens'] < samples['max_num_seqs'], 'max_num_seqs'] = samples['max_num_batched_tokens']
    return samples


def update_max_num_batched_tokens(row, max_sequence_length=4096):
    # If enable_chunked_prefill is False, max_num_batched_tokens must be equal to or greater than max_sequence_length
    if 'enable_chunked_prefill' not in row:
        return row
    enable_chunked_prefill = bool(row.get('enable_chunked_prefill', False))
    if not enable_chunked_prefill and row['max_num_batched_tokens'] < max_sequence_length:
        row['max_num_batched_tokens'] = max_sequence_length
    # Return the updated row
    return row


def update_max_model_len(row):
    if 'max_model_len' not in row or 'max_num_batched_tokens' not in row:
        return row
    max_model_len = row['max_model_len']
    max_num_batched_tokens = row['max_num_batched_tokens']
    if max_model_len is None or max_num_batched_tokens is None:
        return row
    if max_model_len == 0 or max_model_len <= max_num_batched_tokens:
        return row
    if max_num_batched_tokens <= 0:
        row['max_model_len'] = 0
        return row
    row['max_model_len'] = 2 ** int(np.floor(np.log2(max_num_batched_tokens)))
    return row


def prefix_chunked_update(row):
    if 'enable_prefix_caching' not in row or 'enable_chunked_prefill' not in row:
        return row
    enable_prefix_caching = bool(row.get('enable_prefix_caching', False))
    enable_chunked_prefill = bool(row.get('enable_chunked_prefill', False))
    if enable_prefix_caching and enable_chunked_prefill:
        # If both enable_prefix_caching and enable_chunked_prefill are True, then randomly select one or both of them to be False
        random_choice = np.random.choice([1, 2], p=[0.67, 0.33])
        if random_choice == 1:
            # Randomly select one to be False
            if np.random.rand() > 0.5:
                row['enable_prefix_caching'] = False
            else:
                row['enable_chunked_prefill'] = False
        else:
            # Select both of them to be False
            row['enable_prefix_caching'] = False
            row['enable_chunked_prefill'] = False

    return row
