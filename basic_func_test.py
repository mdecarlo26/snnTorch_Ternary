import torch
import torch.nn as nn
import snntorch as snn
import snntorch_ternary
from snntorch import spikegen


beta = 0.9
tlif = snn.TernaryLeaky(beta=beta,threshold=1.0,neg_threshold=1.0)
lif = snn.Leaky(beta=beta,threshold=1.0)

# print("TernaryLeaky neuron:")
# batch=2, features=3
x = torch.tensor([[1, -1, 0],
                  [1, -1, 1]], dtype=torch.float32)

T=3

print("Input tensor:")
print(x)
spk = spikegen.ternary_rate(x, num_steps=T)  # shape: [5, 2, 3]
orig_spk = spikegen.rate(x, num_steps=T)  # shape: [5, 2, 3]
print("Ternary rate-coded spikes:")
for t in range(spk.size(0)):
    print(f"Input Spikes. Step {t}:")
    print(spk[t])
    print("Original rate-coded spikes:")
    print(orig_spk[t])
    s, m = tlif(spk[t])
    sl, ml = lif(orig_spk[t])
    print("T Membrane potential:")
    print(m)
    print("T Output spikes:")
    print(s)
    print("L Membrane potential:")
    print(ml)
    print("L Output spikes:")
    print(sl)
    print("-----")
