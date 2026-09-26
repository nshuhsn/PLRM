import torch
import torch.distributed as dist


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def global_mean(numerator, count):
    count = torch.as_tensor(count, dtype=numerator.dtype, device=numerator.device)
    if world_size() == 1:
        return numerator / count.clamp_min(1)
    totals = torch.cat([numerator.detach().reshape(-1), count.detach().reshape(1)])
    dist.all_reduce(totals)
    denominator = totals[-1].clamp_min(1)
    local = numerator * world_size() / denominator
    value = totals[:-1].reshape_as(numerator) / denominator
    return local + (value - local.detach())


class _ReplicatedBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        size = torch.tensor([tensor.size(0)], device=tensor.device, dtype=torch.long)
        sizes = [torch.empty_like(size) for _ in range(world_size())]
        dist.all_gather(sizes, size)
        lengths = [int(item.item()) for item in sizes]
        rank = dist.get_rank()
        ctx.start = sum(lengths[:rank])
        ctx.length = lengths[rank]
        ctx.world_size = world_size()
        padding = tensor.new_zeros((max(lengths) - tensor.size(0), *tensor.shape[1:]))
        padded = torch.cat([tensor, padding], dim=0).contiguous()
        parts = [torch.empty_like(padded) for _ in lengths]
        dist.all_gather(parts, padded)
        return torch.cat([part[:length] for part, length in zip(parts, lengths)], dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.narrow(0, ctx.start, ctx.length) * ctx.world_size


def gather_batch(tensor, with_grad=True):
    if world_size() == 1:
        return tensor
    if not with_grad:
        tensor = tensor.detach()
    return _ReplicatedBatch.apply(tensor)


def shared_directions(directions):
    if world_size() > 1:
        dist.broadcast(directions, src=0)
    return directions
