"""
Decoder-Only Transformer

I followed along with Andrej Karpathy's "Let's build GPT from scratch" video.

References:
- Andrej video: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2408s
- nanoGPT repo: https://github.com/karpathy/nanoGPT
- Original paper: "Attention Is All You Need" (2017)
"""

import numpy as np

# Model Config
# These control how big and complex the model is.
# I kept them small so it runs fast and is easy to debug

vocab_size = 65       # how many unique tokens (characters) the model knows
n_embd = 64           # the size of each token's vector representation
n_head = 4            # how many attention heads run in parallel
n_layer = 4           # how many transformer blocks we stack
block_size = 32       # max number of tokens the model can look at at once
head_size = n_embd // n_head  # each head gets an equal slice of the embedding
np.random.seed(42) # seed

# HELPER FUNCS
def softmax(x, axis=-1):
    """
    turns raw scores into probabilities (0 to 1, sums to 1). subtract max to avoid overflow.
    """
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def relu(x):
    """
    keeps positives, zeros out negatives. adds nonlinearity so the model can learn complex patterns.
    """
    return np.maximum(0, x)


def layer_norm(x, gamma, beta, eps=1e-5):
    """
    normalizes each token's vector so values don't explode or vanish through layers.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


# Weight Initialization

def init_linear(fan_in, fan_out):
    """ 
    creates a linear layer (matrix multiply + bias). He initialization keeps signal stable.
    """
    scale = np.sqrt(2.0 / fan_in)
    W = np.random.randn(fan_in, fan_out) * scale
    b = np.zeros(fan_out)
    return W, b


def init_layer_norm(dim):
    """
    gamma=1 and beta=0 so initially it just normalizes without changing anything.
    """
    gamma = np.ones(dim)
    beta = np.zeros(dim)
    return gamma, beta


def init_head(n_embd, head_size):
    """
    each attention head needs Query, Key, Value projections.    
    """
    Wq, bq = init_linear(n_embd, head_size)
    Wk, bk = init_linear(n_embd, head_size)
    Wv, bv = init_linear(n_embd, head_size)
    return {'Wq': Wq, 'bq': bq, 'Wk': Wk, 'bk': bk, 'Wv': Wv, 'bv': bv}


def init_block(n_embd, n_head, head_size):
    """
    One transformer block has:
    - multiple attention heads + output projection to combine them
    - a feed forward network (two linear layers with relu)
    - two layer norms (one before attention, one before feed forward)
    """
    heads = [init_head(n_embd, head_size) for _ in range(n_head)]
    Wo, bo = init_linear(n_embd, n_embd)
    ln1 = init_layer_norm(n_embd)
    # feed forward expands to 4x then compresses back, this is standard from the paper
    Wff1, bff1 = init_linear(n_embd, 4 * n_embd)
    Wff2, bff2 = init_linear(4 * n_embd, n_embd)
    ln2 = init_layer_norm(n_embd)
    return {
        'heads': heads, 'Wo': Wo, 'bo': bo,
        'ln1': ln1,
        'Wff1': Wff1, 'bff1': bff1,
        'Wff2': Wff2, 'bff2': bff2,
        'ln2': ln2,
    }


def init_model():
    """
    Putting the whole model together:
    - token_emb: lookup table that converts each token id into a vector
    - pos_emb: lookup table that encodes position (so model knows word order)
    - blocks: the main transformer layers stacked on top of each other
    - ln_f: one final layer norm at the end
    - Wout: projects from embedding space back to vocab size for predictions
    """
    token_emb = np.random.randn(vocab_size, n_embd) * 0.02
    pos_emb = np.random.randn(block_size, n_embd) * 0.02
    blocks = [init_block(n_embd, n_head, head_size) for _ in range(n_layer)]
    ln_f = init_layer_norm(n_embd)
    Wout, bout = init_linear(n_embd, vocab_size)
    return {
        'token_emb': token_emb, 'pos_emb': pos_emb,
        'blocks': blocks, 'ln_f': ln_f,
        'Wout': Wout, 'bout': bout,
    }


# Forward Pass

def attention(x, head_params):
    """
    single head of self-attention. Tokens look at other tokens and decide who is relevant.
    """
    T, C = x.shape  # T = number of tokens, C = embedding size

    # each token produces a query, key, and value
    q = x @ head_params['Wq'] + head_params['bq']  # (T, head_size)
    k = x @ head_params['Wk'] + head_params['bk']  # (T, head_size)
    v = x @ head_params['Wv'] + head_params['bv']  # (T, head_size)

    # how much does each token care about every other token?
    # dividing by sqrt(head_size) so the dot products don't get too big
    scores = (q @ k.T) / np.sqrt(head_size)  # (T, T)

    # this is the causal mask - it prevents tokens from looking into the future
    # we set future positions to -inf so softmax turns them into 0
    mask = np.triu(np.ones((T, T)), k=1) * (-1e9)
    scores = scores + mask

    # convert scores to probabilities
    weights = softmax(scores, axis=-1)  # (T, T)

    # weighted sum of values - each token collects info from tokens it attended to
    out = weights @ v  # (T, head_size)
    return out


def multi_head_attention(x, block):
    """
    Run all attention heads in parallel, then combine their results.
    Each head can learn to focus on different relationships in the data.
    """
    # run each head independently
    heads_out = [attention(x, h) for h in block['heads']]

    # stick all the head outputs together side by side
    concat = np.concatenate(heads_out, axis=-1)  # (T, n_embd)

    # one more linear layer to mix info across heads
    out = concat @ block['Wo'] + block['bo']  # (T, n_embd)
    return out


def feed_forward(x, block):
    """
    two-layer network applied to each token independently. expand 4x, relu, compress back.
    """
    h = relu(x @ block['Wff1'] + block['bff1'])  # (T, 4*n_embd)
    out = h @ block['Wff2'] + block['bff2']       # (T, n_embd)
    return out


def transformer_block(x, block):
    """
    one full block: pre-norm attention with residual, then pre-norm feed-forward with residual.
    """
    # attention part: normalize, attend, add back to input
    ln1_g, ln1_b = block['ln1']
    x = x + multi_head_attention(layer_norm(x, ln1_g, ln1_b), block)

    # feed forward part: normalize, process, add back to input
    ln2_g, ln2_b = block['ln2']
    x = x + feed_forward(layer_norm(x, ln2_g, ln2_b), block)
    return x


def forward(tokens, model):
    """
    complete forward pass - takes token ids in, outputs prediction scores.
    """
    T = len(tokens)

    # step 1: convert token ids to vectors and add positional info
    # so the model knows both WHAT each token is and WHERE it is
    x = model['token_emb'][tokens] + model['pos_emb'][:T]  # (T, n_embd)

    # step 2: pass through each transformer block
    # each block lets tokens communicate (attention) then think (feed forward)
    for block in model['blocks']:
        x = transformer_block(x, block)

    # step 3: final layer norm
    ln_g, ln_b = model['ln_f']
    x = layer_norm(x, ln_g, ln_b)

    # step 4: project to vocab size to get a score for each possible next token
    logits = x @ model['Wout'] + model['bout']  # (T, vocab_size)
    return logits


if __name__ == "__main__":
    model = init_model()
    # just feeding in random tokens to show the forward pass works
    tokens = np.random.randint(0, vocab_size, size=block_size)
    logits = forward(tokens, model)
    print(f"Input tokens shape:  {tokens.shape}")
    print(f"Output logits shape: {logits.shape}")
