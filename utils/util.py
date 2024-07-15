import torch
import torch.nn.functional as F


def get_next_word(logits, temp=None, k=None, p=None, greedy=None, m=None):
    probs = F.softmax(logits, dim=-1)
    logprobs = F.log_softmax(logits, dim=-1)

    if temp is not None:
        samp_probs = F.softmax(logits.div_(temp), dim=-1)
    else:
        samp_probs = probs.clone()

    if greedy:
        next_probs, next_tokens = probs.topk(1)
        if next_tokens.shape[0] == 1:
            next_tokens = next_tokens.unsqueeze(0)
            logprobs = logprobs.unsqueeze(0)
        next_logprobs = logprobs.gather(1, next_tokens.view(-1, 1))

    elif k is not None:
        indices_to_remove = samp_probs < torch.topk(samp_probs, k)[0][..., -1, None]
        samp_probs[indices_to_remove] = 0
        if m is not None:
            samp_probs.div_(samp_probs.sum(1).unsqueeze(1))
            samp_probs.mul_(1-m)
            samp_probs.add_(probs.mul(m))
        next_tokens = samp_probs.multinomial(1)
        next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()

    elif p is not None:
        sorted_probs, sorted_indices = torch.sort(samp_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        sorted_samp_probs = sorted_probs.clone()
        sorted_samp_probs[sorted_indices_to_remove] = 0
        if m is not None:
            sorted_samp_probs.div_(sorted_samp_probs.sum(1).unsqueeze(1))
            sorted_samp_probs.mul_(1-m)
            sorted_samp_probs.add_(sorted_probs.mul(m))
        sorted_next_indices = sorted_samp_probs.multinomial(1).view(-1, 1)
        next_tokens = sorted_indices.gather(1, sorted_next_indices)
        next_logprobs = sorted_samp_probs.gather(1, sorted_next_indices).log()

    else:
        if m is not None:
            samp_probs.div_(samp_probs.sum(1).unsqueeze(1))
            samp_probs.mul_(1-m)
            samp_probs.add_(probs.mul(m))
        next_tokens = samp_probs.multinomial(1)
        next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()
    return next_tokens.squeeze(1), next_logprobs
