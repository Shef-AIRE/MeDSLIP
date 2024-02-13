import torch.distributed as dist

dist.init_process_group(backend='nccl', init_method='env://localhost:5678')
print(dist.get_rank(), dist.get_world_size())