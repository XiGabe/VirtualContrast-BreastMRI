"""One trainer for NC, CE and VCE; validation selects the checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, log_loss

from .classifier import Classifier
from .data import validate_cache, read_manifest, BreastDataset, BalancedSampler, collate
from .inference import predict
from .utils import seed_all, sha256, save_checkpoint, write_json, write_csv, rng_state, restore_rng


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('configs/default.json'))
    p.add_argument('--manifest', type=Path, default=Path('reproducibility/cohort.csv'))
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--modality', choices=['nc', 'ce', 'vce'], required=True)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--dinov2-weights', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--chunk-size', type=int)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())['classifier']
    for key in ('batch_size', 'chunk_size'):
        if getattr(args, key):
            cfg[key] = getattr(args, key)
    micro, effective = cfg['batch_size'], cfg['effective_batch']
    if micro < 1 or effective % micro or cfg['chunk_size'] < 1 or cfg['epochs'] < 1:
        raise ValueError('Invalid training configuration')
    device = torch.device(args.device)
    torch.set_num_threads(4)
    seed_all(args.seed)
    rows = read_manifest(args.manifest)
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    if any({r['label'] for r in group} != {'0','1'} for group in (train, val)):
        raise ValueError('Train and validation must contain both classes')
    config = dict(settings=cfg, seed=args.seed, modality=args.modality, manifest_sha256=sha256(args.manifest),
                  cache_signature=validate_cache(train + val, args.cache, [args.modality]))
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / 'config.json'
    if args.resume:
        if json.loads(marker.read_text()) != config:
            raise ValueError('Resume configuration changed')
        if not (args.output / 'best.pt').exists():
            raise FileNotFoundError('Resume requires best.pt')
    elif marker.exists():
        raise FileExistsError('Existing run; use --resume or another output directory')
    write_json(marker, config)
    model = Classifier(args.dinov2_weights, pretrained=not args.resume,
                       chunk_size=cfg['chunk_size'], dropout=cfg['dropout']).to(device)
    optimizer = torch.optim.AdamW([
        dict(params=model.encoder.parameters(), lr=cfg['encoder_lr']),
        dict(params=[p for n,p in model.named_parameters() if not n.startswith('encoder.')], lr=cfg['head_lr'])
    ], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
        factor=cfg['lr_factor'], patience=cfg['lr_patience'], threshold=cfg['min_delta'], threshold_mode='abs',
        min_lr=[cfg['encoder_lr']*cfg['min_lr_ratio'], cfg['head_lr']*cfg['min_lr_ratio']], eps=0.)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    start, best_auc, best_bce, reference, stale, history = 0, -1., float('inf'), -1., 0, []
    if args.resume:
        state = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        start = state['epoch']+1
        best_auc, best_bce, reference, stale = (state[k] for k in ('best_auc', 'best_bce', 'reference', 'stale'))
        history = state['history']
        restore_rng(state['rng'])
    sampler = BalancedSampler(train, args.seed)
    loader = DataLoader(BreastDataset(train, args.cache, args.modality, True), batch_size=micro,
                        sampler=sampler, num_workers=args.workers, collate_fn=collate, pin_memory=device.type == 'cuda')
    validation = DataLoader(BreastDataset(val, args.cache, args.modality), batch_size=micro,
                            num_workers=args.workers, collate_fn=collate)
    for epoch in range(start, cfg['epochs']):
        if stale >= cfg['patience']:
            break
        sampler.epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, seen, skipped = 0., 0, 0
        for i, (volumes, labels, _) in enumerate(loader):
            labels = labels.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                loss = F.binary_cross_entropy_with_logits(model(volumes), labels, reduction='sum')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite classifier loss')
            group_n = min(effective, len(train)-(i // (effective//micro))*effective)
            scaler.scale(loss / group_n).backward()
            if (i+1) % (effective//micro) == 0 or i+1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < scale)
                optimizer.zero_grad(set_to_none=True)
            total += loss.item()
            seen += len(labels)
        predictions = predict(model, validation, device)
        labels = np.array([r['label'] for r in predictions])
        logits = np.array([r['logit'] for r in predictions])
        # Compute validation probabilities in float32.
        probabilities = torch.tensor(logits, dtype=torch.float32).sigmoid().numpy()
        auc = roc_auc_score(labels, probabilities)
        bce = float(log_loss(labels, probabilities.astype(np.float64), labels=[0,1]))
        improved = auc > best_auc or (auc == best_auc and bce < best_bce)
        if improved:
            best_auc, best_bce = auc, bce
        if auc > reference + cfg['min_delta']:
            reference, stale = auc, 0
        else:
            stale += 1
        scheduler.step(auc)
        history.append(dict(epoch=epoch, train_bce=total/seen, val_auroc=auc, val_bce=bce,
                            skipped_steps=skipped, stale=stale))
        light = dict(format=1, kind='classifier', model=model.state_dict(), config=config,
                     modality=args.modality, seed=args.seed, epoch=epoch)
        if improved:
            save_checkpoint(args.output / 'best.pt', light)
            write_csv(args.output / 'val_predictions.csv', predictions)
        save_checkpoint(args.output / 'last.pt', dict(**light, optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
            best_auc=best_auc, best_bce=best_bce, reference=reference, stale=stale, history=history))
        write_json(args.output / 'history.json', history)
        print(json.dumps(history[-1]), flush=True)


if __name__ == '__main__':
    main()
