# coding=utf-8
from TempNormalization import *
from krippendorff_alpha import *

from collections import defaultdict

dic1, dic2 = defaultdict(dict), defaultdict(dict)
out1, out2 = [], []

with open('/Users/fei-c/Resources/timex/Event Time Corpus/event-times-annotator1_normalized.tab', 'r') as fi:
    for line in fi:
        toks = line.strip().split()
        dic1[toks[0]][toks[4]] = toks[-1]
        # dic1[toks[0]][toks[4]] = normalize_anchor(toks[-1]) if toks[-1] else None

with open('/Users/fei-c/Resources/timex/Event Time Corpus/event-times-annotator2_normalized.tab', 'r') as fi:
    for line in fi:
        toks = line.strip().split()
        dic2[toks[0]][toks[4]] = toks[-1]
        # dic2[toks[0]][toks[4]] = normalize_anchor(toks[-1]) if toks[-1] else None


agree, total = 0, 0

for doc_id, doc in dic1.items():
    for event_id, time in doc.items():
        total += 1
        out1.append(dic1[doc_id][event_id])
        out2.append(dic2[doc_id][event_id])
        if dic1[doc_id][event_id] == dic2[doc_id][event_id]:
            agree += 1
        else:
            print(dic1[doc_id][event_id], " ||| ", dic2[doc_id][event_id])



print(agree, total, agree/total)
print("kappa: ", krippendorff_alpha([out1, out2],
                                    nominal_metric,
                                    convert_items=str,
                                    missing_items='n/a'))



