"""
naivebayes.py
~~~~~~~~~~
This script demonstrates a simple Naive Bayes Classifier of text.
Reference: Group exercise from CS124 Language to Information at Stanford (taught by D. Jurafsky)

Yejin Cho (scarletcho@gmail.com)
Last updated: 2016-11-17
"""

import re
# ------------------------------------------------------------------------------------
# Data preparation: Load corpus data for train & test
train = open('training','r')
trainset = train.readlines()
train.close()

# ------------------------------------------------------------------------------------
# (1) Compute prior for two classes (+), (-)
poscnt_bysent = 0
negcnt_bysent = 0

for sent in trainset:
    if re.search('\+', sent):
        poscnt_bysent += 1
    elif re.search('-', sent):
        negcnt_bysent += 1

prior_pos = poscnt_bysent / float(len(trainset))
prior_neg = negcnt_bysent / float(len(trainset))

print('\n\n< STEP1: PRIOR modeled by my NAIVE BAYES CLASSIFIER>')
print('Prior for POSITIVE class: '+ str(prior_pos))
print('Prior for NEGATIVE class: '+ str(prior_neg))

# ------------------------------------------------------------------------------------
# (2) Compute likelihood for all words in training set given the class
items = []
labs = []

# Split into items (by words) and labels
for i in range(0, len(trainset)):
    splitted = trainset[i].split()
    items.extend(splitted[1:])
    labs.extend(splitted[0] * len(splitted[1:]))

# Get unique list of items
unique_list = list(set(items))
unique_like_pos = []
unique_like_neg = []
smoothing_alpha = 1.0

for word in unique_list:
    poscnt_byword = 0
    negcnt_byword = 0

    for k in range(0, len(items)):
        if re.match(word, items[k]):
            if re.match('\+', labs[k]):
                poscnt_byword += 1
            elif re.match('-', labs[k]):
                negcnt_byword += 1

    unique_like_pos.append( (poscnt_byword + smoothing_alpha) / (len(unique_list) + smoothing_alpha*len(items)) )
    unique_like_neg.append( (negcnt_byword + smoothing_alpha) / (len(unique_list) + smoothing_alpha*len(items)) )

unique_like_pos_print = ['POS likelihood: ' + '{:.3f}'.format(x) for x in unique_like_pos]
unique_like_neg_print = ['NEG likelihood: ' + '{:.3f}'.format(x) for x in unique_like_neg]

print('\n\n< STEP2: LIKELIHOOD modeled by my NAIVE BAYES CLASSIFIER>')
for pair in zip(unique_list, unique_like_pos_print, unique_like_neg_print):
    print '\n>> '.join(pair) + '\n'

# ------------------------------------------------------------------------------------
# (3) Compute whether the sentence in the test set is of class positive or negative
test = open('test','r')
testset = test.readlines()
test.close()

testset = testset[0].split()
test_like_pos_list = []
test_like_neg_list = []

for i in range(0, len(testset)):
    testitem = testset[i]
    for k in range(0, len(unique_list)):
        if re.match(testitem, unique_list[k]):
            test_like_pos_list.append(unique_like_pos[k])
            test_like_neg_list.append(unique_like_neg[k])

test_like_pos = reduce(lambda x, y: x*y, test_like_pos_list)
test_like_neg = reduce(lambda x, y: x*y, test_like_neg_list)

posterior_pos = prior_pos * test_like_pos
posterior_neg = prior_neg * test_like_neg

posterior_pos_print = 'POS posterior: ' + '{:.4f}'.format(posterior_pos)
posterior_neg_print = 'NEG posterior: ' + '{:.4f}'.format(posterior_neg)

print('\n< STEP3: TEST RESULT >')
print(posterior_pos_print + '\n' + posterior_neg_print)

if posterior_pos > posterior_neg:
    print('=> [Prediction] more likely to be POSITIVE')
else:
    print('=> [Prediction] more likely to be NEGATIVE')
