"""Train generator for the fixed study budget; final epoch is the endpoint."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import validate_cache, read_manifest, GeneratorDataset
from .generator import Generator
from .losses import generator_loss
from .utils import seed_all, sha256, save_checkpoint, write_json, rng_state, restore_rng


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/default.json'))
    p.add_argument('--manifest', type=Path, default=Path('reproducibility/cohort.csv'))
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())['generator']
    if args.batch_size:
        cfg['batch_size'] = args.batch_size
    micro, effective = cfg['batch_size'], cfg['effective_batch']
    if micro < 1 or effective % micro or cfg['epochs'] < 1:
        raise ValueError('Invalid batch size or epochs')
    device = torch.device(args.device)
    seed_all(cfg['seed'])
    torch.set_num_threads(4)
    rows = [r for r in read_manifest(args.manifest) if r['split'] == 'train']
    config = dict(settings=cfg, manifest_sha256=sha256(args.manifest),
                  cache_signature=validate_cache(rows, args.cache, ['nc','ce','breast']))
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / 'config.json'
    if args.resume:
        if json.loads(marker.read_text()) != config:
            raise ValueError('Resume configuration changed')
    elif marker.exists():
        raise FileExistsError('Existing run; use --resume or another output directory')
    write_json(marker, config)
    model = Generator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    start, history = 0, []
    if args.resume:
        state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        opt.load_state_dict(state['optimizer'])
        start, history = state['epoch'] + 1, state['history']
        restore_rng(state['rng'])
    dataset = GeneratorDataset(rows, args.cache, cfg['patch'], cfg['seed'])
    for epoch in range(start, cfg['epochs']):
        dataset.epoch = epoch
        lr = cfg['min_lr'] + .5 * (cfg['lr']-cfg['min_lr']) * (1+math.cos(math.pi*epoch/max(cfg['epochs']-1,1)))
        for group in opt.param_groups:
            group['lr'] = lr
        options = dict(num_workers=args.workers, pin_memory=device.type == 'cuda')
        if args.workers:
            options.update(multiprocessing_context='spawn', prefetch_factor=1)
        loader = DataLoader(dataset, batch_size=micro, shuffle=True,
                            generator=torch.Generator().manual_seed(cfg['seed']+epoch), **options)
        total, seen, updates = 0., 0, 0
        model.train()
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                delta = model(batch['pre'])
            loss, _ = generator_loss(delta, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite generator loss')
            n = len(batch['pre'])
            group_n = min(effective, len(rows)-(i // (effective//micro))*effective)
            (loss * n / group_n).backward()
            if (i+1) % (effective//micro) == 0 or i+1 == len(loader):
                if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
                    raise FloatingPointError('Nonfinite generator gradient')
                opt.step()
                opt.zero_grad(set_to_none=True)
                updates += 1
            total += loss.item()*n
            seen += n
        history.append(dict(epoch=epoch, loss=total/seen, lr=lr, updates=updates))
        state = dict(format=1, kind='generator', model=model.state_dict(), config=config,
                     epoch=epoch, optimizer=opt.state_dict(), rng=rng_state(), history=history)
        save_checkpoint(args.output / 'last.pt', state)
        write_json(args.output / 'history.json', history)
        print(json.dumps(history[-1]), flush=True)
    # Save the final epoch.
    save_checkpoint(args.output / 'generator.pt', dict(format=1, kind='generator',
        model=model.state_dict(), seed=cfg['seed'], epoch=cfg['epochs']-1, config=config))


if __name__ == '__main__':
    main()
