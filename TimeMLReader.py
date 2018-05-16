import xml.etree.ElementTree as ET
import os, sys
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.stanford import StanfordTokenizer

from Mention import *
from TempLink import *

def load_timeml(filename):
    root = ET.parse(filename).getroot()
    doc = None

    ## read DOCID and DCT information
    for elem in root.iter():
        if elem.tag == "DOCID":
            doc = TimeMLDoc(docid=elem.text)
        if elem.tag == "DCT":
            for step_elem in elem:
                doc.dct = Timex(content=step_elem.text, **step_elem.attrib)
    if not doc:
        raise Exception("%s cannot find docid information" % filename)

    ## read text information
    for elem in root.iter():

        if elem.tag == 'TEXT':
            if elem.text:
                toks = elem.text.split()
                doc.appendTokens(toks)
            for step_elem in elem:
                toks = step_elem.text.split()
                tok_ids = [ len(doc.tokens) + i for i in range(len(toks))]
                doc.appendTokens(toks)
                if step_elem.tag == "EVENT":
                    doc.addEvent(Event(content=' '.join(toks), tok_ids=tok_ids, **step_elem.attrib))
                elif step_elem.tag == "TIMEX3":
                    doc.addTimex(Timex(content=' '.join(toks), tok_ids=tok_ids, **step_elem.attrib))
                elif step_elem.tag == "SIGNAL":
                    doc.addSignal(Signal(content=' '.join(toks), tok_ids=tok_ids, **step_elem.attrib))
                else:
                    raise Exception("Unknown mention type: %s" % (step_elem.tag))
                if step_elem.tail:
                    toks = step_elem.tail.split()
                    doc.appendTokens(toks)
    return doc

def batch_load(file_dir):
    doc_list = []
    for filename in os.listdir(file_dir):
        if filename.endswith('.tml'):
            fullname = os.path.join(file_dir, filename)
            doc_list.append(load_timeml(fullname))
    return doc_list

def save_batch_load(file_dir, out_file):
    doc_list = batch_load(file_dir)
    with open(out_file, 'w',encoding="us-ascii") as fo:
        for doc in doc_list:
            for event in doc.events:
                fo.write("%s\t%s\t%s\n" % (doc.docid, event.eid, event.tanchor))
    return doc_list

def load_tanchor_num(in_file):
    doc_dic = {}
    with open(in_file, 'r', encoding="us-ascii") as fi:
        for line in fi:
            docid = line.strip().split()[0]
            doc_dic.setdefault(docid, 0)
            doc_dic[docid] += 1
    return doc_dic

def load_tanchor(in_file):
    doc_dic = {}
    with open(in_file, 'r', encoding="us-ascii") as fi:
        for line in fi:
            toks = line.strip().split()
            docid = toks[0]
            eid = toks[4]
            tanchor = toks[7]
            doc_dic.setdefault(docid, {})
            doc_dic[docid][eid] = tanchor
    return doc_dic

def compare_tanchors(tanchor1, tanchor2):
    if not tanchor1 or not tanchor2:
        raise TypeError
    out1 = tanchor1
    out2 = tanchor2old(tanchor2)
    return 1 if out1 == out2 else 0

def uncertain2old(anchor):
    out = anchor.strip().strip("()")
    # print(out)
    out = out.replace(',', '').replace(' ', '')
    return out

def tanchor2old(anchor):
    name_map = {'begin':'beginPoint', 'end':'endPoint', 'nd':'endPoint'}
    if ':' not in anchor:
        return uncertain2old(anchor)
    else:
        out = ""
        anchor = anchor.strip().strip("()")
        toks = anchor.split(', f')[0].split(', e')
        for tok in toks:
            key, value = tok.strip().split(':')
            out += "%s=%s" % (name_map[key], uncertain2old(value))
        return out



import unittest

class TestTimeMLReader(unittest.TestCase):

    def test_load_timeml(self):
        filename = "/Users/fei-c/Resources/timex/TimeAnchor2/DNS/DNS001_ABC19980108.1830.0711.tml"
        doc = load_timeml(filename)
        print(doc.docid, doc.dct.tid, doc.dct.category, doc.dct.content, doc.dct.value)
        print(len(doc.tokens), len(doc.events), len(doc.timexs), len(doc.signals))
        print(len(sent_tokenize(' '.join(doc.tokens))))

    def test_batch_load(self):
        file_dir = "/Users/fei-c/Resources/timex/TimeAnchor2/DNS"
        batch_load(file_dir)

    def test_save_batch_load(self):
        file_dir = "/Users/fei-c/Resources/timex/TimeAnchor2/DNS"
        out_file = "tanchor.txt"
        save_batch_load(file_dir, out_file)

    def test_compare_annotations(self):

        class_dic = {'I_ACTION': [0, 0], 'OCCURRENCE': [0, 0], 'PERCEPTION': [0, 0], 'REPORTING': [0, 0], 'ASPECTUAL': [0, 0], 'I_STATE': [0, 0], 'STATE': [0, 0], 'ALL':[0, 0]}
        file_dir = "/Users/fei-c/Resources/timex/TimeAnchor2/DNS"
        out_file = "tanchor.txt"
        tanchor_file = "/Users/fei-c/Resources/timex/Event Time Corpus/event-times_normalized.tab"
        doc_list = save_batch_load(file_dir, out_file)
        doc_num = load_tanchor_num(tanchor_file)
        for doc in doc_list:
            print(doc.docid, len(doc.events), doc_num[doc.docid])
        doc_dic = load_tanchor(tanchor_file)
        print(len(doc_dic))
        for doc in doc_list:
            for e in doc.events:
                try:
                    tanchor1 = doc_dic[doc.docid][e.eid]
                    tanchor2 = e.tanchor
                    if e.eclass == 'ASPECTUAL':
                        print(doc.docid, e.content, e.eclass, e.eid, ' | ', e.tanchor, ' | ', doc_dic[doc.docid][e.eid], ' | ', compare_tanchors(tanchor1, tanchor2))
                    class_dic[e.eclass][0] += compare_tanchors(tanchor1, tanchor2)
                    class_dic[e.eclass][1] += 1
                    class_dic['ALL'][0] += compare_tanchors(tanchor1, tanchor2)
                    class_dic['ALL'][1] += 1
                except Exception as ex:
                    print('Error:', doc.docid, e.eid, e.content, e.eclass, e.tanchor)
        for key, (c, a) in class_dic.items():
            print("%s / %.2f / %s" % (key, c / a if a != 0 else 0, a))

    def test_compare_tanchors(self):
        tanchor2 = "(begin:(after 1947-02-01, before 1947-02-28), end:after 1998-02-27)"
        tanchor1 = "beginPoint=after1947-02-01before1947-02-28endPoint=after1998-02-27"
        # tanchor2 = "(after 1990-08-12, before 1990-08-16)"
        # tanchor1 = "after1990-08-12before1990-08-16"
        print(compare_tanchors(tanchor1, tanchor2))