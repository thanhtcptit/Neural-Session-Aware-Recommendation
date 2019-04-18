import os
from utils.qpath import SUMMARY_DIR

filename = 'UserAGRU-pre-t-step_loss.txt'


avg_loss = 0.
batch_num = 0
beta = 0.8
losses = []

with open(os.path.join(SUMMARY_DIR, filename)) as f:
    for loss in f:
        batch_num += 1
        loss = float(loss)
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        losses.append(str(smoothed_loss).replace('.', ','))

with open(os.path.join(SUMMARY_DIR, filename + '-s'), 'w') as f:
    f.write('\n'.join(losses))
