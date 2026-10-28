"""Generator training objective."""
import math
import torch

def _masked_mean_per_case(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    numerator = (values * mask).flatten(1).sum(1)
    denominator = mask.flatten(1).sum(1)
    result = numerator / denominator.clamp_min(1)
    return torch.where(denominator > 0, result, torch.zeros_like(result))

def generation_loss(
    predicted_delta: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    valid: torch.Tensor,
    breast: torch.Tensor,
    guidance: torch.Tensor,
    boundary_confidence: torch.Tensor,
    breast_weight: float,
    outside_weight: float,
    enhancement_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Three independently normalized objectives; padding contributes to none."""
    predicted_post = pre + predicted_delta
    reconstruction_error = (predicted_post - post).abs()
    delta_error = (predicted_delta - (post - pre)).abs()
    breast_region = valid.bool() & breast.bool()
    outside_region = valid.bool() & ~breast.bool()
    enhancement_weights = guidance * boundary_confidence * breast_region.to(guidance.dtype)

    breast_l1 = _masked_mean_per_case(reconstruction_error, breast_region).mean()
    outside_l1 = _masked_mean_per_case(reconstruction_error, outside_region).mean()
    enhancement_l1 = _masked_mean_per_case(delta_error, enhancement_weights).mean()
    total = (
        breast_weight * breast_l1
        + outside_weight * outside_l1
        + enhancement_weight * enhancement_l1
    )
    return total, {
        "breast_l1": breast_l1,
        "outside_l1": outside_l1,
        "enhancement_l1": enhancement_l1,
    }
def excess_enhancement_loss(delta, pre, post, valid, breast, boundary,
                            margin=0.02, fraction=0.05):
    """Mean of the largest excess-enhancement errors per case.
    """
    if margin < 0 or not 0 < fraction <= 1:
        raise ValueError('Invalid excess-enhancement margin/fraction')
    error = torch.relu(torch.relu(delta.float())
                       - torch.relu(post.float() - pre.float()) - margin)
    region = valid.bool() & breast.bool() & (boundary.float() >= 0.5)
    losses = []
    for e, mask in zip(error, region, strict=True):
        values = e[mask]
        if values.numel():
            k = max(1, math.ceil(values.numel() * fraction))
            losses.append(values.topk(k, sorted=False).values.mean())
        else:
            losses.append(e.sum() * 0.)
    return torch.stack(losses).mean()
def missed_enhancement_loss(predicted_delta, pre, post, q, confidence, margin=.01):
    if margin<0: raise ValueError('Negative margin')
    w=q.float()*confidence.float()
    error=torch.relu(post.float()-pre.float()-predicted_delta.float()-margin)
    return ((error*w).flatten(1).sum(1)/w.flatten(1).sum(1).clamp_min(1)).mean()

def generator_loss(delta, batch):
    delta = delta.float()
    base, parts = generation_loss(delta, batch['pre'], batch['post'], batch['valid'],
        batch['breast'], batch['guidance'], batch['boundary'], 1., .1, 2.)
    extra = excess_enhancement_loss(delta, batch['pre'], batch['post'], batch['valid'],
        batch['breast'], batch['boundary'])
    confidence = (batch['boundary'] >= .5).float() * batch['valid'] * batch['breast']
    missing = missed_enhancement_loss(delta, batch['pre'], batch['post'], batch['target'], confidence)
    return base + .03 * extra + missing, dict(**parts, excess=extra, missing=missing)
