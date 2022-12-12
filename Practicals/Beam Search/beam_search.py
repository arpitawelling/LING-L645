import json
from math import log
import numpy as np
import matplotlib.pyplot as plt

# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 0.0]]
    # walk over each step in sequence

    max_T, max_A = data.shape

    # Loop over time
    for t in range(max_T):
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            # Loop over possible alphabet outputs
            for c in range(max_A - 1):
                candidate = [seq + [c], score - log(data[t, c])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences

def removeDuplicates(s):
    if len(s) < 2:
        return s
    if s[0] != s[1]:
        return s[0]+removeDuplicates(s[1:])
    return removeDuplicates(s[1:])
# define a sequence of 10 words (rows) over a vocab of 5 words (columns), 
# e.g.
#      a  bites cat  dog  the
# 1   0.1  0.2  0.3  0.4  0.5
# 2   0.5  0.3  0.5  0.2  0.1
# ...
# 10  0.3  0.4  0.5  0.2  0.1 

# data = [[0.1, 0.2, 0.3, 0.4, 0.5],
#         [0.4, 0.3, 0.5, 0.2, 0.1],
#         [0.1, 0.2, 0.3, 0.4, 0.5],
#         [0.5, 0.4, 0.3, 0.2, 0.1],
#         [0.1, 0.2, 0.3, 0.4, 0.5],
#         [0.5, 0.4, 0.3, 0.2, 0.1],
#         [0.1, 0.2, 0.3, 0.4, 0.5],
#         [0.5, 0.4, 0.3, 0.2, 0.1],
#         [0.1, 0.2, 0.3, 0.4, 0.5],
#         [0.3, 0.4, 0.5, 0.2, 0.1]]

output = open('output.json')
output_acoustic = json.load(output)

data = output_acoustic['logits']
#print(data)
data = np.array(data)

beam_width = 3

# decode sequence
result = beam_search_decoder(data, beam_width)
# print result
for i, seq in enumerate(result):
    print(i, seq)

alphabet = output_acoustic['alphabet'] 
#print(alphabet)

outputs = []
for i in result:
    chars = i[0]
    sent = []
    for j in chars:
        sent.append(alphabet[j])
    outputs.append(''.join(sent))

#print(outputs)
outputs2=[]
for i in outputs:
    s = removeDuplicates(i)
    outputs2.append(s)

print(outputs2)
fig, ax = plt.subplots()
im = ax.imshow(data)
plt.show()