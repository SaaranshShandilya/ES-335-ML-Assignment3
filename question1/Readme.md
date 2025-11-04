## Comparitive Analysis

# Dataset size
Category 1:
Length of the vocabulary is: 20082
Number of samples: 588090
Category 2:
Length of the vocabulary is: 30274
Number of samples: 884049

# Summary
The primary difference in between two categories was for pre processing of data. In the case of natural languages, a single word was deemed to be a token but in the case of code recommendation, our tokens even included punctuation marks and other literals. Morever, for category 1, the loss was limited to 5.9 whereas in category 2 loss went down uptil 3.2 before the model started to overfit. This was because of a diverse dataset and a bigger dataset used for next token prediction for c++ language. Embedding which were learnt were better as compared to previous to category 1.

