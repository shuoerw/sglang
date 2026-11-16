"""Hierarchical CPU all-gather for small scheduler metadata tensors."""

from functools import lru_cache

import torch.distributed as dist


def _same_node_members(group, source_rank):
    from sglang.srt.distributed.parallel_state import in_the_same_node_as

    return in_the_same_node_as(group, source_rank=source_rank)


@lru_cache(maxsize=None)
def _hierarchy(group):
    """Build groups once; fall back for nonuniform or interleaved layouts.

    All members of the parent group must call this collectively. Subgroup
    ranks are global ranks, while topology discovery uses parent-group ranks.
    """
    world = dist.get_world_size(group)
    first_node = _same_node_members(group, 0)
    local_size = sum(first_node)
    if (
        local_size <= 1
        or local_size >= world
        or world % local_size
        or first_node != [i < local_size for i in range(world)]
    ):
        return None

    for start in range(local_size, world, local_size):
        members = _same_node_members(group, start)
        if members != [start <= i < start + local_size for i in range(world)]:
            return None

    parent_ranks = dist.get_process_group_ranks(group)
    rank = dist.get_rank(group)
    node_group = None
    for start in range(0, world, local_size):
        ranks = parent_ranks[start : start + local_size]
        subgroup = dist.new_group(ranks=ranks, backend="gloo")
        if start <= rank < start + local_size:
            node_group = subgroup
    leaders = parent_ranks[::local_size]
    leader_group = dist.new_group(ranks=leaders, backend="gloo")
    return (
        node_group,
        leader_group,
        local_size,
        parent_ranks[(rank // local_size) * local_size],
        rank % local_size == 0,
    )


def try_hierarchical_all_gather(global_flat, local_row, group):
    """Gather rows in group-rank order, or return False to use the default path.

    The parent group must span WORLD, with uniform, contiguous node membership.
    Node-local gather, leader all-gather, and node-local broadcast preserve
    exactly the same output layout as a flat all-gather.
    """
    if local_row.device.type != "cpu" or dist.get_backend(group) != "gloo":
        return False
    world = dist.get_world_size(group)
    if world <= 2:
        return False
    # Group creation must be called by every WORLD rank. Subsets keep the
    # default exchange instead of creating groups from the scheduler hot path.
    if dist.get_process_group_ranks(group) != list(range(dist.get_world_size())):
        return False
    hierarchy = _hierarchy(group)
    if hierarchy is None:
        return False

    node_group, leader_group, local_size, leader_rank, is_leader = hierarchy
    node_rows = local_row.new_empty(local_size * local_row.numel())
    dist.all_gather_into_tensor(
        node_rows, local_row.contiguous().view(-1), group=node_group
    )
    if is_leader:
        dist.all_gather_into_tensor(global_flat, node_rows, group=leader_group)
    dist.broadcast(global_flat, src=leader_rank, group=node_group)
    return True
