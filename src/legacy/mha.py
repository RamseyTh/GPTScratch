import torch
import math

'''
Complete this module such that it computes queries, keys, and values,
computes attention, and passes through a final linear operation W_o.

You do NOT need to apply a causal mask (we will do that next week).
If you don't know what that is, don't worry, we will cover it next lecture.

Be careful with your tensor shapes! Print them out and try feeding data through
your model. Make sure it behaves as you would expect.
'''
class CustomMHA(torch.nn.Module):
    '''
    param d_model : (int) the length of vectors used in this model
    param n_heads : (int) the number of attention heads. You can assume that
        this even divides d_model.
    '''
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # W_qkv: (3D, D), W_o: (D, D)
        self.W_qkv = torch.nn.Parameter(torch.randn(3 * d_model, d_model))
        self.W_o   = torch.nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x):
        
        B, S, D = x.shape
        h = self.n_heads
        d_h = self.d_head

        # use W_qkv to get queries Q, keys K, values V, each of shape (B,S,D)
        qkv = x @ self.W_qkv.T  
        Q, K, V = qkv.split(D, dim=-1)  # each (B, S, D)

        # reshape these into size (B, h, S, D/h)
        Q = Q.view(B, S, h, d_h).transpose(1, 2)
        K = K.view(B, S, h, d_h).transpose(1, 2)
        V = V.view(B, S, h, d_h).transpose(1, 2)

        # compute QK^T and divide by sqrt(D/h)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_h)

        # causal Mask
        causal_mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # apply softmax 
        attn = torch.softmax(scores, dim=-1)

        # matrix multiply against values
        # reshape (B,h,S,D/h) into (B,S,D)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, S, D)

        # matrix multiply against output projection W_o
        return out @ self.W_o.T

if __name__ == "__main__":

	# example of building and running this class
	mha = CustomMHA(128,8)

	# 32 samples of length 6 each, with d_model at 128
	x = torch.randn((32,6,128))
	y = mha(x)
	print(x.shape, y.shape) # should be the same