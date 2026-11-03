"""Supplementary frozen configurations using the unchanged domain producer.

The shared fine sparse representation is also granted to source-wise lifting.
This wrapper adds its own source to every snapshot and source-drift check;
the original driver is imported without changing its file or run logic.
"""
from pathlib import Path

import benchmark_policy_domains as b


WRAPPER_PATH = Path(__file__).resolve()
_original_source_paths = b.source_paths
_original_method_order = b.method_order
_original_protocol = b.protocol
b.SPECS = b.SPECS + (
    b.Spec('lift_sparse_fine_preview', mode='sparse', certifier='lifting',
        atom_tol=1e-6, keep=512, preview_keep=128, depth_override=4),
)


def source_paths():
    # Keeping the wrapper under ROOT/scripts makes producer relative paths
    # valid without changing how the original producer freezes other files.
    WRAPPER_PATH.relative_to(b.ROOT)
    return sorted(set(_original_source_paths()) | {WRAPPER_PATH})


b.source_paths = source_paths


def method_order(specs, qi, block):
    # The legacy reverse-and-rotate rule cancels for two methods. Give each
    # matched method alternating first position; preserve every other rule.
    if len(specs) == 2:
        return list(specs) if (qi+block) % 2 == 0 else list(reversed(specs))
    return _original_method_order(specs, qi, block)


def protocol(*args, **kwargs):
    result = _original_protocol(*args, **kwargs)
    result['method_order_policy'] = 'two_alternating_else_original'
    result['query_order'] = ('Unchanged original query prefix. With exactly two methods, '
        'use original spec order on even(qi+block), reversed spec order on odd(qi+block). '
        'All other method counts retain the original reverse-on-odd then cyclic-rotate rule.')
    return result


b.method_order = method_order
b.protocol = protocol


if __name__ == '__main__':
    b.main()
