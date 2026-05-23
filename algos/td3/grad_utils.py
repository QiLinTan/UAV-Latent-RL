def clamp_encoder_grad_scale(scale: float) -> float:
    return max(0.0, min(1.0, float(scale)))


def resolve_encoder_grad_scale(scale, legacy_updates_encoder: bool, default: float) -> float:
    if scale is None:
        scale = 1.0 if legacy_updates_encoder else default
    return clamp_encoder_grad_scale(scale)


def parse_encoder_grad_schedule(spec):
    if not spec:
        return None

    schedule = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        step_str, scale_str = item.split(":", 1)
        schedule.append((max(0, int(step_str)), clamp_encoder_grad_scale(float(scale_str))))

    if not schedule:
        return None
    return tuple(sorted(schedule, key=lambda pair: pair[0]))


def scheduled_encoder_grad_scale(default_scale: float, schedule, step: int) -> float:
    scale = clamp_encoder_grad_scale(default_scale)
    if not schedule:
        return scale

    step = max(0, int(step))
    for milestone_step, milestone_scale in schedule:
        if step < milestone_step:
            break
        scale = milestone_scale
    return clamp_encoder_grad_scale(scale)


def soft_detach(tensor, grad_scale: float):
    grad_scale = clamp_encoder_grad_scale(grad_scale)
    if grad_scale <= 0.0:
        return tensor.detach()
    if grad_scale >= 1.0:
        return tensor
    detached = tensor.detach()
    return detached + grad_scale * (tensor - detached)
