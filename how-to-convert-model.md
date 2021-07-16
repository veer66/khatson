# How to convert Attacut model to PyTorch JIT format

1. Install attacut

```
pip install attacut
```

2. Run this script

```Python
import torch
from attacut import Tokenizer

tok = Tokenizer('attacut-c')
txt = "กาก้ากูกู้"
tokens, features = tok.dataset.make_feature(txt)
inputs = (features, torch.Tensor(0))
x, _, _ = tok.dataset.prepare_model_inputs(inputs)
print(x)
m = torch.jit.trace(tok.model, (x,))
m.save("attacut-c.pt")
```
