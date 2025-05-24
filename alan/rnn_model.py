import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SwingGRU(nn.Module):
    def __init__(self, input_dim = 6, d_model = 32, output_dim = 4, dropout=0.2, weight=None):
        super(SwingGRU, self).__init__()
        self.numerical_linear  = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model)
            )
        
        self.rnn = nn.GRU(d_model, d_model,
                            num_layers = 2, 
                            batch_first=True,
                            bidirectional=False,
                            dropout=dropout,)

        self.linear_out  = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_dim),
        )

        self.criterion = nn.CrossEntropyLoss(weight = weight)
        # self._reinitialize()
            
    def forward(self, x, target = None):
        
        x = self.numerical_linear(x)
        output, _ = self.rnn(x)
        last = output[:, -1, :]
        logits = self.linear_out(last)
        
        if target is None:
            return logits, last
        else:
            loss = self.criterion(logits, target)
            return logits, loss
        
    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        use_fused = False
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


if __name__ == "__main__":
    model = SwingGRU(weight=torch.tensor([1.0, 1.0, 1.0, 1.0]))
    input = torch.randn(32, 10, 6)  # batch_size=32, seq_len=10, input_dim=6
    target = torch.randint(0, 4, (32,))  # batch_size=32, output_dim=4
    loss, output = model(input, target)
    print("Loss:", loss.item())
    print("Output shape:", output.shape)
