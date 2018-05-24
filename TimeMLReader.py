import xml.etree.ElementTree as ET
import os, sys, pickle
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.stanford import StanfordTokenizer

from TempMention import Token, Timex, Event, Signal
from TempLink import TimeMLDoc, TempLink
import TempUtils



def load_extraml(file_name):
    root = ET.parse(file_name).getroot()
    doc = None

    ## read DOCID and DCT information
    for elem in root.iter():
        if elem.tag == "DOCID" or "DOCNO":
            doc = TimeMLDoc(docid=elem.text)
        if elem.tag == "DCT":
            for step_elem in elem:
                doc.dct = Timex(content=step_elem.text, **step_elem.attrib)
    if not doc:
        raise Exception("%s cannot find docid information" % file_name)

    ## read text information
    for elem in root.iter():
        if elem.tag == 'TEXT':
            sent_id = 0
            for sent_elem in elem:
                if sent_elem.text:
                    toks = sent_elem.text.split()
                    for i in range(len(toks)):
                        tokc = Token(content=toks[i], tok_id=len(doc.tokens)+i, sent_id=sent_id)
                        doc.tokens.append(tokc)              
                for mention_elem in sent_elem:
                    toks = mention_elem.text.split()
                    tok_ids = [len(doc.tokens) + i for i in range(len(toks))]
                    for i in range(len(toks)):
                        tokc = Token(content=toks[i], tok_id=len(doc.tokens)+i, sent_id=sent_id)
                        doc.tokens.append(tokc)
                    if mention_elem.tag == "EVENT":
                        doc.addEvent(Event(content=' '.join(toks), tok_ids=tok_ids, sent_id=sent_id, **mention_elem.attrib))
                    elif mention_elem.tag == "TIMEX3":
                        doc.addTimex(Timex(content=' '.join(toks), tok_ids=tok_ids, sent_id=sent_id, **mention_elem.attrib))
                    elif mention_elem.tag == "SIGNAL":
                        doc.addSignal(Signal(content=' '.join(toks), tok_ids=tok_ids, sent_id=sent_id, **mention_elem.attrib))
                    else:
                        raise Exception("Unknown mention type: %s" % (step_elem.tag))
                    if mention_elem.tail:
                        toks = mention_elem.tail.split()
                        for i in range(len(toks)):
                            tokc = Token(content=toks[i], tok_id=len(doc.tokens)+i, sent_id=sent_id)
                            doc.tokens.append(tokc)
                sent_id += 1
    return doc


def load_timeml(file_name):
    root = ET.parse(file_name).getroot()
    doc = None

    ## read DOCID and DCT information
    for elem in root.iter():
        if elem.tag == "DOCID":
            doc = TimeMLDoc(docid=elem.text)
        if elem.tag == "DCT":
            for step_elem in elem:
                doc.dct = Timex(content=step_elem.text, **step_elem.attrib)
                doc.addTimex(Timex(content=step_elem.text, **step_elem.attrib))
    if not doc:
        raise Exception("%s cannot find docid information" % file_name)

    ## read text information
    for elem in root.iter():

        if elem.tag == 'TEXT':
            if elem.text:
                toks = elem.text.split()
                for i in range(len(toks)):
                    tokc = Token(content=toks[i], tok_id=len(doc.tokens) + i)
                    doc.tokens.append(tokc)
            for step_elem in elem:
                toks = step_elem.text.split()
                tok_ids = [ len(doc.tokens) + i for i in range(len(toks))]
                for i in range(len(toks)):
                    tokc = Token(content=toks[i], tok_id=len(doc.tokens) + i)
                    doc.tokens.append(tokc)
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
                    for i in range(len(toks)):
                        tokc = Token(content=toks[i], tok_id=len(doc.tokens)+i)
                        doc.tokens.append(tokc)
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


def load_TBD(file_name):
    rel_dic = {}
    with open(file_name, 'r', encoding='utf-8') as fi:
        for line in fi:
            toks = line.strip().split()
            rel_dic.setdefault(toks[0], [])
            rel_dic[toks[0]].append([toks[1], toks[2], toks[3]])
    # for key, rels in rel_dic.items():
    #     print(key, len(rels))
    # print('count:', sum([len(rels) for rels in rel_dic.values()]))
    return rel_dic


def prepare_rels(rel_file, timeml_dir):
    rel_dic = load_TBD(rel_file)
    doc_list = {}
    for full_name in os.listdir(timeml_dir):
        doc_id, extention = os.path.splitext(full_name)
        if doc_id in rel_dic.keys():
            doc_path = os.path.join(timeml_dir, full_name)
            doc_list[doc_id] = load_timeml(doc_path)
    for doc_id, rels in rel_dic.items():
        doc = doc_list[doc_id]
        for sour, targ, rel in rels:
            try:
                words = word_tokenize(' '.join(doc.geneInterTokens(sour, targ)))
                temprel = TempLink(sour=doc.getMentionById(sour), targ=doc.getMentionById(targ), rel=rel)
                temprel.interwords = words
                temprel.interpos = TempUtils.geneInterPostion(words)
                doc.addTlink(temprel)
            except Exception as ex:
                # print(doc_id, rel)
                # print(ex)
                pass

    return doc_list


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

    def test_load_extra(self):
        doc = load_extraml("/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL044_wsj_0173.tml")
#        print([ tok.content for tok in doc.tokens])
        event1 = doc.events[0]
        event2 = doc.events[1]
        print(event1.content, event1.tok_ids)
        print(event2.content, event2.tok_ids)
        print(doc.geneInterTokens(event1, event2))
        print([tok.content for tok in doc.tokens[:25]])
        print(doc.geneInterPostion(event1, event2))

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
                    # pass
                    print('Error:', doc.docid, e.eid, e.content, e.eclass, e.tanchor)
        for key, (c, a) in class_dic.items():
            print("%s / %.2f / %s" % (key, c / a if a != 0 else 0, a))

    def test_compare_tanchors(self):
        tanchor2 = "(begin:(after 1947-02-01, before 1947-02-28), end:after 1998-02-27)"
        tanchor1 = "beginPoint=after1947-02-01before1947-02-28endPoint=after1998-02-27"
        # tanchor2 = "(after 1990-08-12, before 1990-08-16)"
        # tanchor1 = "after1990-08-12before1990-08-16"
        print(compare_tanchors(tanchor1, tanchor2))

    def test_prepare_rels(self):
        rel_file = 'data/TimebankDense.T3.txt'
        timeml_dir = '/Users/fei-c/Resources/timex/TBAQ-cleaned.bak/TimeBank'
        # rel_file = 'data/temporalorder.txt'
        doc_list = prepare_rels(rel_file, timeml_dir)
        print(sum([ len(doc.tlinks) for doc_id, doc in doc_list.items()]))
        with open('data/doc_list.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(doc_list, f)
