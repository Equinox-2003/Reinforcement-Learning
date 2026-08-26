import numpy as np

def sample(dices=2):
    x = 0
    for _ in range(dices):
        x += np.random.choice([1, 2, 3, 4, 5, 6])
    return x

trival = 1000
V = n = 0

for _ in range(trival):
    s = sample()
    n += 1
    # nv = (s1 + s2 + ... + sn) / n = v * (n-1)/n + sn/n
    # v += (s-v)/n
    V += (s - V) / n
    print(V)