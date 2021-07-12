import sys
import os

import numpy as np

data_fn = sys.argv[1]
data = np.load(data_fn, allow_pickle=True).T

rseed = 7324
np.random.seed(rseed)
shuffle = np.arange(0, data.shape[1])
np.random.shuffle(shuffle)
data = data[:, shuffle]

outdn = './PS1_data'
if not os.path.exists(outdn):
    os.makedirs(outdn)

X = np.array(data[4:21], dtype=np.float32)
X[-1] = np.log(X[-1]) # E(B-V)

zerr = data[3].astype(np.float32)
zspec = data[2].astype(np.float32)
zspec[zspec < 0] = 0
labels = np.zeros(len(zspec))
Y = np.vstack((labels, zspec, zerr)).astype(np.float32).T

minX = np.min(X, axis=1).reshape(-1, 1)
maxX = np.max(X, axis=1).reshape(-1, 1)

print(minX)
print(maxX)
quit()
normedX = (X-minX)/(maxX-minX)*2.-1.

normed = np.hstack((Y, normedX.T.astype(np.float32)))

# Data ratio for training, validation
dtrain_ratio = 0.8
dval_ratio = 0.1

dlen = len(normed)
itrain = int(dlen*dtrain_ratio)
ival = int(dlen*dval_ratio)

dtrain = normed[:itrain]
dval = normed[itrain:itrain+ival]
dtest = normed[itrain+ival:]
outputs = [dtrain, dval, dtest]

phases = ['train', 'val', 'test']
for i, phase in enumerate(phases):
    fsave = os.path.join(outdn, '%s.npy' % phases[i])
np.save(fsave, outputs[i])
