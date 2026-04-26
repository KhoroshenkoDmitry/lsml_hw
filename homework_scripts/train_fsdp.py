#!/usr/bin/env -S uv run --script

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import tqdm
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    default_data_collator,
)

from common import LocalTimer, get_mem_stats, load_and_preprocess_data, rank0_first

LOGGER = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment-name', default=None)
    parser.add_argument('-d', '--dataset-name', default=None, required=True)
    parser.add_argument('-ds', '--dataset-subset', default=None)
    parser.add_argument('-m', '--model-name', default=None, required=True)
    parser.add_argument('--save-dir', default='outputs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num-epochs', default=100, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--ckpt-freq', default=500, type=int)
    parser.add_argument('-s', '--seq-length', default=1024, type=int)
    parser.add_argument('--dtype', default='bf16', choices=['fp32', 'bf16'])
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--grad-accum-steps', default=1, type=int)
    parser.add_argument(
        '--sharding-strategy',
        choices=['no_shard', 'shard_grad_op', 'full_shard'],
        required=True,
        help='no_shard=DDP, shard_grad_op=FSDP2 reshard_after_forward=False, full_shard=FSDP2 reshard_after_forward=True',
    )
    parser.add_argument(
        '--no-compile',
        action='store_true',
        help='Disable torch.compile (useful if FSDP2 + compile glitches)',
    )
    return parser


def wrap_model_for_strategy(
    model: torch.nn.Module,
    strategy: str,
    mesh: Any,
    local_rank: int,
) -> torch.nn.Module:
    """Apply the requested sharding strategy and return the (possibly wrapped) model."""
    if strategy == 'no_shard':
        # Pure DDP — full replication of params/grads/optim states.
        return DistributedDataParallel(
            model,
            device_ids=[local_rank],
            bucket_cap_mb=500,
            gradient_as_bucket_view=True,
        )

    # FSDP2 path. Shard each transformer block + the root.
    # For pythia (GPTNeoX) the layers live at model.gpt_neox.layers.
    # If you swap to another arch, adjust this attribute path.
    reshard_after_forward = strategy == 'full_shard'

    # Try common attribute paths — be defensive across HF model variants.
    transformer_layers = None
    for attr_path in (
        ('gpt_neox', 'layers'),
        ('transformer', 'h'),
        ('model', 'layers'),
    ):
        obj = model
        try:
            for a in attr_path:
                obj = getattr(obj, a)
            transformer_layers = obj
            break
        except AttributeError:
            continue

    if transformer_layers is None:
        raise RuntimeError(
            'Could not locate transformer layers on the model; '
            'add an attribute path for this architecture.'
        )

    for layer in transformer_layers:
        fully_shard(layer, mesh=mesh, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
    return model


@record
def main(args: argparse.Namespace) -> None:  # noqa: C901, PLR0915, PLR0912
    rank = int(os.getenv('RANK', '0'))
    local_rank = rank % torch.cuda.device_count()
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(rank=rank, world_size=world_size, device_id=device)

    logging.basicConfig(
        format=f'[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s',
        level=logging.INFO,
    )

    LOGGER.debug(os.environ)
    LOGGER.debug(args)
    LOGGER.debug(f'local_rank={local_rank} rank={rank} world_size={world_size}')
    LOGGER.info(f'Sharding strategy: {args.sharding_strategy}')

    dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16}[args.dtype]
    torch.manual_seed(args.seed)

    # Initializing an **untrained** model
    model: torch.nn.Module
    with rank0_first(), device:
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    LOGGER.info(f'Training {sum(p.numel() for p in model.parameters())} model parameters')

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    if not args.no_compile:
        model = torch.compile(model)  # type: ignore[assignment]

    LOGGER.info(f'Initialized model uses {get_mem_stats(device)["curr_alloc_gb"]}gb')

    # Build a 1D mesh across all ranks for FSDP2.
    mesh = init_device_mesh('cuda', (world_size,))
    model = wrap_model_for_strategy(model, args.sharding_strategy, mesh, local_rank)

    LOGGER.info(f'After wrap: model uses {get_mem_stats(device)["curr_alloc_gb"]}gb')

    with rank0_first():
        data = load_and_preprocess_data(
            args.model_name,
            args.seq_length,
            args.dataset_name,
            args.dataset_subset,
            config,
        )
    data = data.train_test_split(test_size=0.05, seed=args.seed)
    train_data = data['train']
    eval_data = data['test']
    LOGGER.debug(f'{len(train_data)} training samples. {len(eval_data)} eval samples')

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=default_data_collator,
        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=default_data_collator,
        sampler=DistributedSampler(eval_data, shuffle=False, drop_last=True),
    )
    LOGGER.info(f'{len(dataloader)} train batches per epoch, {len(eval_dataloader)} eval batches per epoch')

    # IMPORTANT: optimizer must be built AFTER the model is wrapped/sharded,
    # otherwise it will hold references to the unsharded parameter tensors.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=args.lr * 1e-2)

    is_experiment = False
    exp_dir: Path = Path(args.save_dir)
    if args.experiment_name is not None:
        is_experiment = True
        exp_dir = exp_dir / args.experiment_name

    state = {
        'epoch': 0,
        'global_step': 0,
        'epoch_step': 0,
        'running_loss': 0,
    }
    # NOTE: resume logic removed for FSDP2 — distributed checkpointing for FSDP2
    # needs torch.distributed.checkpoint, which is out of scope here. If you need
    # resume, add it via dcp.save / dcp.load on a state_dict context manager.

    dist.barrier()
    if is_experiment and rank == 0:
        LOGGER.info('Creating experiment root directory')
        exp_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    timers = {k: LocalTimer(device) for k in ['data', 'forward', 'backward', 'update']}

    for state['epoch'] in range(state['epoch'], args.num_epochs):  # noqa: B020
        LOGGER.info(f'Begin epoch {state["epoch"]} at step {state["epoch_step"]}')
        model.train()

        progress_bar = tqdm.tqdm(range(len(dataloader)), disable=(rank != 0))
        if state['epoch_step'] > 0:
            progress_bar.update(state['epoch_step'])

        dataloader.sampler.set_epoch(state['epoch'])  # type: ignore[attr-defined]
        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            with timers['data'], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            if i_step < state['epoch_step']:
                continue

            with timers['forward']:
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum_steps
                del batch

            with timers['backward']:
                loss.backward()

            is_accum_boundary = (i_step + 1) % args.grad_accum_steps == 0

            with timers['update']:
                if is_accum_boundary:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            state['epoch_step'] += 1
            state['running_loss'] += loss.item() * args.grad_accum_steps
            progress_bar.update(1)

            if not is_accum_boundary:
                continue

            state['global_step'] += 1

            if state['global_step'] % args.log_freq == 0:
                # Per-step token count is global: every rank consumes batch_size*seq_len,
                # multiplied by world_size, multiplied by grad_accum_steps.
                tok_per_step = args.batch_size * args.seq_length * world_size * args.grad_accum_steps
                ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
                info = {
                    'global_step': state['global_step'],
                    'lr': lr_scheduler.get_last_lr()[0],
                    'running_loss': state['running_loss'] / args.log_freq,
                    'epoch': state['epoch'],
                    'epoch_progress': state['epoch_step'] / len(dataloader),
                    'num_batches_remaining': len(dataloader) - i_step,
                    **get_mem_stats(device),
                    'tokens_per_s': 1000 * tok_per_step / ms_per_step,
                    'time/total': ms_per_step,
                    **{f'time/{k}': timer.avg_elapsed_ms() for k, timer in timers.items()},
                }

                LOGGER.info(info)

                torch.cuda.reset_peak_memory_stats(device)
                state['running_loss'] = 0
                for t in timers.values():
                    t.reset()

            # Skipping checkpointing for FSDP2 — see note above.
            # For DDP (no_shard) we could still save, but keep behavior uniform.

        # ---------- Eval ----------
        # Important: for FSDP2 (FULL_SHARD especially) you MUST run forward on
        # all ranks so the framework can all-gather params. Don't gate eval on rank==0.
        dist.barrier()
        model.eval()
        local_loss_sum = torch.zeros(1, device=device)
        local_count = torch.zeros(1, device=device)

        for batch in eval_dataloader:
            batch = {k: v.to(device=device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            local_loss_sum += outputs.loss.detach().float()
            local_count += 1

        # Average loss across all ranks (each rank saw a different shard of eval data).
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
        eval_loss = (local_loss_sum / local_count).item()

        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float('inf')

        if rank == 0:
            LOGGER.info(f'epoch {state["epoch"]}: perplexity: {perplexity} eval_loss: {eval_loss}')

        dist.barrier()
        state['epoch_step'] = 0

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)