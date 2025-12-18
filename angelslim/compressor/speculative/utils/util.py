import random
from collections import deque

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


class MomentumScorePredictor:
    def __init__(self, window_size=10):
        if window_size < 2:
            raise ValueError("Window size must be at least 2")
        self.scores = deque(maxlen=window_size)
        self.delta_window_size = window_size - 1
        self.deltas = deque(maxlen=self.delta_window_size)
        self.last_score = None

    def add_score(self, score):
        self.scores.append(score)
        if len(self.scores) > 1:
            delta = score - self.scores[-2]
            self.deltas.append(delta)
        self.last_score = score

    def predict_next_score(self):
        if len(self.deltas) < self.delta_window_size:
            return None
        average_delta = sum(self.deltas) / len(self.deltas)
        return self.last_score + average_delta


class EWMAScorePredictor:
    def __init__(self, alpha=0.2, initial_score=None):
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        self.alpha = alpha
        self.ewma = initial_score

    def add_score(self, score):
        if self.ewma is None:
            self.ewma = score
        else:
            self.ewma = self.alpha * score + (1 - self.alpha) * self.ewma

    def predict_next_score(self):
        return self.ewma


class MeanScorePredictor:
    def __init__(self, window_size=100):
        self.scores = deque(maxlen=window_size)

    def add_score(self, score):
        self.scores.append(score)

    def predict_next_score(self):
        if len(self.scores) == 0:
            return None
        return sum(self.scores) / len(self.scores)

    def clear_before(self):
        if len(self.scores) == 0:
            return
        score = self.scores[-1]
        self.scores.clear()
        self.scores.append(score)


def prepare_logits_processor(
    temperature: float = 0.0,
    repetition_penalty: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def evaluate_posterior(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    logits_processor,
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided
    logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or
    evaluates posterior probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor):
        Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float):
        Softmax temperature for probability scaling. A value of 0 indicates
        greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - sample_token (torch.Tensor): Target model recover token of the best candidate.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position
        # in the sequence
        posterior_mask = (candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        sample_p = logits[best_candidate, accept_length]
        sample_token = torch.argmax(sample_p)
        sample_token = sample_token[None, None]
        return best_candidate, accept_length, sample_token

    else:
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = 1.0
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        gtp[xi] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            sample_p = torch.softmax(gt_logits, dim=0)
        sample_token = torch.multinomial(sample_p, 1)
        sample_token = sample_token[None]
        return torch.tensor(best_candidate), accept_length - 1, sample_token


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:, ...])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1, ...]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:, ...], zeropadding), dim=1)
    return tensor
