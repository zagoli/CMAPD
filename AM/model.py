import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from AM.environment import Collective


class Embedder(nn.Module):
    def __init__(self, input_size, d_model):
        super().__init__()
        self.is_continuous = ~torch.tensor([bool(s) for s in input_size], dtype=torch.bool)

        size = input_size.count(0)
        self.continuous_embedding = nn.Linear(size, d_model) if size > 0 else None
        self.categorical_embedding = nn.ModuleList([
            nn.Embedding(1 + s, d_model, padding_idx=0)
            for s in input_size if s != 0])

        d_continuous = d_model if size > 0 else 0
        self.output = nn.Linear(d_continuous + d_model, d_model)

    def forward(self, x):
        continuous = x[..., self.is_continuous].float()
        categorical = x[..., ~self.is_continuous].long()

        categorical_embedded = torch.stack([
            emb(1 + categorical[..., i])
            for i, emb in enumerate(self.categorical_embedding)], dim=-1).sum(dim=-1)

        if self.continuous_embedding is not None:
            continuous_embedded = self.continuous_embedding(continuous)
            return self.output(torch.cat([continuous_embedded, categorical_embedded], dim=-1))

        return self.output(categorical_embedded)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None, output_attention=False):
        batch_size = q.size(0)

        q = self.w_q(q).view(batch_size, self.nhead, -1, self.d_head)
        k = self.w_k(k).view(batch_size, self.nhead, -1, self.d_head)
        v = self.w_v(v).view(batch_size, self.nhead, -1, self.d_head)

        u = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if output_attention:
            return u.mean(dim=1)

        if mask is not None:
            u = u.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(u, dim=-1)

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), 0)

        attn_applied = torch.matmul(attn, v)
        return self.w_o(attn_applied.view(batch_size, -1, self.d_model))


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.attn = nn.ModuleList([
            MultiHeadAttention(d_model, nhead) for _ in range(num_layers)])

        self.ffrd = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, d_model))
            for _ in range(num_layers)])

        self.norm1 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.norm2 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for attn, ffrd, norm1, norm2 in zip(self.attn, self.ffrd, self.norm1, self.norm2):
            x = norm1(x + attn(x, x, x, mask))
            x = norm2(x + ffrd(x))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers=1):
        super().__init__()
        self.attn = nn.ModuleList([
            MultiHeadAttention(d_model, nhead) for _ in range(num_layers)])

        self.ffrd = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 512),
                nn.ReLU(),
                nn.Linear(512, d_model))
            for _ in range(num_layers)])

        self.norm1 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.norm2 = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.output = MultiHeadAttention(d_model, 1)

    def forward(self, x, h, mask=None):
        for attn, ffrd, norm1, norm2 in zip(self.attn, self.ffrd, self.norm1, self.norm2):
            h = norm1(h + attn(h, x, x, mask))
            h = norm2(h + ffrd(h))

        return self.output(h, x, x, output_attention=True).squeeze(1)


def _get_hidden_state(context, indices):
    batch_size, _, h = context.size()
    ctx = torch.cat([torch.zeros(batch_size, 1, h, device=context.device), context], dim=1)
    ctx = ctx.unsqueeze(1).expand([-1, indices.size(1), -1, -1])
    return ctx.gather(2, 1 + indices.unsqueeze(-1).expand([-1, -1, -1, h])).mean(dim=2)


def _get_mask(tasks, indices):
    batch_size, size_t, _ = tasks.size()
    batch_size, size_a, size_c = indices.size()
    mask = torch.zeros(batch_size, size_a, 1 + size_t, dtype=torch.bool, device=indices.device)
    src = torch.ones(batch_size, size_a, size_c, dtype=torch.bool, device=indices.device)
    mask.scatter_add_(2, 1 + indices, src)
    return mask[..., 1:].any(dim=1, keepdim=True)


class Transformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.ff = nn.Linear(d_model, d_model)
        self.embedder_tasks = Embedder(input_size, d_model)
        self.encoder_tasks = Encoder(d_model, nhead, dim_feedforward, num_layers)

        self.embedder_agents = Embedder(input_size[:2], d_model)
        self.encoder_agents = Encoder(d_model, nhead, dim_feedforward, num_layers)

        self.decoder = Decoder(d_model, nhead)

        self.v = nn.Parameter(torch.empty(1, 1, d_model).uniform_(
            -1 / math.sqrt(d_model), 1 / math.sqrt(d_model)))

    def forward(self, tasks, assignment: Collective):
        """
        Input:
            - tasks: Pytorch tensor of tasks representation [batch_size, max_number_of_tasks, 4]
            - assignment: Collective class inside environment.py
                - indices: Pytorch tensor of indices [batch_size, number_of_agents, max_collective_size]
                - paths: Pytorch tensor of paths representation [batch_size, number_of_agents, X]
        Output:
            - probability: Pytorch tensor [batch_size, number_of_agents, number_of_tasks]
        """
        padding_mask = torch.all((tasks == -1), dim=-1).unsqueeze(
            1)  # Masking pad tokens in the tasks (Because of variable size tasks)
        assignment_mask = _get_mask(tasks,
                                    assignment.indices)  # Masking indices of tasks already picked in the assignmemnt
        full_mask = (assignment.indices != -1).all(dim=-1,
                                                   keepdim=True)  # Masking assignments which are already full (Because of max collective size)
        mask = (assignment_mask | padding_mask | full_mask).transpose(-1, -2)  # Combining the three masks

        h_t = self.embedder_tasks(tasks)  # Obtain hidden representation of the tasks with an embedding layer
        h_t = self.encoder_tasks(h_t,
                                padding_mask)  # Obtain hidden representation of the tasks with an attention-based encoder

        h_s = _get_hidden_state(h_t,
                               assignment.indices)  # Obtain hidden representation of the assignment by combining the hidden representation of tasks in the current assignment

        a_features = torch.cat((assignment.agents, assignment.paths), dim=2)

        h_a = self.embedder_agents(
            a_features)  # Obtain hidden representation of the agents with an embedding layer
        h_a = self.encoder_agents(h_a)  # Obtain hidden representation of the agents with an attention-based encoder

        # hp = self.embedder_paths(assignment.paths)                             # Obtain hidden representation of the paths with an embedding layer (NOT IMPLEMENTED)
        # hp = self.encoder_paths(hp)                                            # Obtain hidden representation of the paths with an attention-based encoder (NOT IMPLEMENTED)

        h_combined = h_s + h_a  # Combine assignment and paths hidden representation (NOT IMPLEMENTED)

        probs = 8 * torch.tanh(self.decoder(h_combined, h_t,
                                            mask))  # Scale the attention weights computed by the decoder with C * tanh(attention_weights) (C = 8 worked well in the past)
        probs = F.softmax(probs.masked_fill(mask, float("-inf")), dim=-1)  # Normalize the probabilities with a softmax
        probs = probs.masked_fill(mask,
                                  0)  # Assign probability 0 to the elements which cannot be selected indicated by the mask

        return probs.transpose(-1, -2)
