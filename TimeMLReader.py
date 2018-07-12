import xml.etree.ElementTree as ET
import os, sys, pickle
import traceback

from nltk import sent_tokenize, word_tokenize
from nltk.tokenize.stanford import StanfordTokenizer

from TempObject import *
from TempOrder import InduceMethod
from TempNormalization import *
import TempUtils


def load_mentions(timeml_file):
    events, timexs, signals = [], [], []
    root = ET.parse(timeml_file).getroot()

    ## read text information
    for elem in root.iter():

        if elem.tag == 'TEXT':
            for mention_elem in elem:
                if mention_elem.tag == "EVENT":
                    events.append(mention_elem.attrib['eid'])
                elif mention_elem.tag == "TIMEX3":
                    timexs.append(mention_elem.attrib['tid'])
                elif mention_elem.tag == "SIGNAL":
                    signals.append(mention_elem.attrib['sid'])

    return events, timexs, signals


def load_anchorml(file_name):
    print('*' * 80)
    print(file_name)
    root = ET.parse(file_name).getroot()
    doc = None

    ## read DOCID and DCT information
    for elem in root.iter():
        if elem.tag in ["DOCID", "DOCNO"]:
            doc = TimeMLDoc(docid=elem.text)
        if elem.tag in ["DCT", "DATE_TIME>"]:
            for mention_elem in elem:
                dct = Timex(content=mention_elem.text, sent_id=0, **mention_elem.attrib)
                doc.dct = dct
    if not doc:
        raise Exception("%s cannot find docid information" % file_name)

    ## read text information
    for elem in root.iter():
        if elem.tag == 'TEXT':
            if elem.text:
                # toks = elem.text.split()
                # for i in range(len(toks)):
                tokenized = word_tokenize(elem.text)
                for j in range(len(tokenized)):
                    tokc = Token(content=tokenized[j], tok_id=len(doc.tokens))
                    doc.tokens.append(tokc)
            for mention_elem in elem:
                # toks = mention_elem.text.split()
                tok_ids = []
                content = []
                # for i in range(len(toks)):
                #     tokenized = word_tokenize(toks[i])
                #     content.extend(tokenized)
                tokenized = word_tokenize(mention_elem.text)
                for j in range(len(tokenized)):
                    tokc = Token(content=tokenized[j], tok_id=len(doc.tokens))
                    content.append(tokenized[j])
                    tok_ids.append(tokc.tok_id)
                    doc.tokens.append(tokc)
                if mention_elem.tag == "EVENT":
                    doc.addEvent(Event(content=' '.join(content), tok_ids=tok_ids, **mention_elem.attrib))
                elif mention_elem.tag == "TIMEX3":
                    doc.addTimex(Timex(content=' '.join(content), tok_ids=tok_ids, **mention_elem.attrib))
                elif mention_elem.tag == "SIGNAL":
                    doc.addSignal(Signal(content=' '.join(content), tok_ids=tok_ids, **mention_elem.attrib))
                else:
                    raise Exception("Unknown mention type: %s" % (mention_elem.tag))
                if mention_elem.tail:
                #     toks = mention_elem.tail.split()
                #     for i in range(len(toks)):
                    tokenized = word_tokenize(mention_elem.tail)
                    for j in range(len(tokenized)):
                        tokc = Token(content=tokenized[j], tok_id=len(doc.tokens))
                        doc.tokens.append(tokc)
    return doc


def load_extraml(extraml_file, events, timexs, signals):
    root = ET.parse(extraml_file).getroot()
    words, sent_ids = [], []

    labs = set([])

    ## read text information
    for elem in root.iter():

        if elem.tag == 'TEXT':
            sent_id = 0
            for sent_elem in elem:

                if sent_elem.tag == 's':
                    text = ""
                    if sent_elem.text:
                        text += sent_elem.text
                    for mention_elem in sent_elem:
                        if mention_elem.tag in ["CARDINAL", "NUMEX", "COMMENT", "ENAMEX"]:
                            if mention_elem.text:
                                text += mention_elem.text
                            for step_elem in mention_elem:
                                text += step_elem.text
                                if step_elem.tail:
                                    text += step_elem.tail
                            if mention_elem.tail:
                                text += mention_elem.tail
                        else:
                            if mention_elem.tag in ["EVENT"]:
                                if mention_elem.attrib['eid'] in events:
                                    text += " %s " % (mention_elem.text)
                                else:
                                    text += mention_elem.text
                                if mention_elem.tail:
                                    text += mention_elem.tail
                            elif mention_elem.tag in ["TIMEX3"]:
                                if mention_elem.attrib['tid'] in timexs:
                                    if mention_elem.text:
                                        text += " %s " % (mention_elem.text)
                                    for step_elem in mention_elem:
                                        if step_elem.text:
                                            text += step_elem.text
                                        for step2_elem in step_elem:
                                            text += step2_elem.text
                                            if step2_elem.tail:
                                                text += step2_elem.tail
                                        if step_elem.tail:
                                            text += "%s " % (step_elem.tail)
                                    if mention_elem.tail:
                                        text += " %s" % (mention_elem.tail)
                                else:
                                    text += mention_elem.text
                                    if mention_elem.tail:
                                        text += mention_elem.tail

                            elif mention_elem.tag in ["SIGNAL"]:
                                if mention_elem.attrib['sid'] in signals:
                                    text += " %s " % (mention_elem.text)
                                else:
                                    text += mention_elem.text
                                if mention_elem.tail:
                                    text += mention_elem.tail
                    for tok in text.split():
                        words.append(tok)
                        sent_ids.append(sent_id)

                    sent_id += 1

                elif sent_elem.tag == 'turn':
                    for s_elem in sent_elem:
                        if s_elem.tag == 's':
                            text = ""
                            if s_elem.text:
                                print(s_elem.text)
                                text += s_elem.text
                            for mention_elem in s_elem:
                                if mention_elem.tag in ["CARDINAL", "NUMEX", "COMMENT", "ENAMEX"]:
                                    if mention_elem.text:
                                        text += mention_elem.text
                                    for step_elem in mention_elem:
                                        text += step_elem.text
                                        if step_elem.tail:
                                            text += step_elem.tail
                                    if mention_elem.tail:
                                        text += mention_elem.tail
                                else:
                                    if mention_elem.tag in ["EVENT"]:
                                        if mention_elem.attrib['eid'] in events:
                                            text += " %s " % (mention_elem.text)
                                        else:
                                            text += mention_elem.text
                                        if mention_elem.tail:
                                            text += mention_elem.tail
                                    elif mention_elem.tag in ["TIMEX3"]:
                                        if mention_elem.attrib['tid'] in timexs:
                                            if mention_elem.text:
                                                text += " %s " % (mention_elem.text)
                                            for step_elem in mention_elem:
                                                if step_elem.text:
                                                    text += step_elem.text
                                                for step2_elem in step_elem:
                                                    text += step2_elem.text
                                                    if step2_elem.tail:
                                                        text += step2_elem.tail
                                                if step_elem.tail:
                                                    text += "%s " % (step_elem.tail)
                                            if mention_elem.tail:
                                                text += " %s" % (mention_elem.tail)
                                        else:
                                            text += mention_elem.text
                                            if mention_elem.tail:
                                                text += mention_elem.tail

                                    elif mention_elem.tag in ["SIGNAL"]:
                                        if mention_elem.attrib['sid'] in signals:
                                            text += " %s " % (mention_elem.text)
                                        else:
                                            text += mention_elem.text
                                        if mention_elem.tail:
                                            text += mention_elem.tail

                            for tok in text.split():
                                words.append(tok)
                                sent_ids.append(sent_id)

                        sent_id += 1
    print(labs)
    return words, sent_ids


def load_anchorml_sentid(anchor_file, extraml_file, verbose=0):

    ## load sent ids information from the original timebank extraml files
    events, timexs, signals = load_mentions(anchor_file)

    words, sent_ids = load_extraml(extraml_file, events, timexs, signals)
    print(len(words), len(sent_ids))

    print("Completing load sentence ids information...", extraml_file)
    word_cps = [] ## for asserting same words to extraml words


    root = ET.parse(anchor_file).getroot()

    doc = None

    ## read DOCID and DCT information
    for elem in root.iter():
        if elem.tag in ["DOCID", "DOCNO"]:
            doc = TimeMLDoc(docid=elem.text)
        if elem.tag in ["DCT", "DATE_TIME>"]:
            for dct_elem in elem:
                dct = Timex(content=dct_elem.text, **dct_elem.attrib)
                doc.dct = dct
                # doc.addTimex(dct)
    if not doc:
        raise Exception("%s cannot find docid information" % file_name)

    ## read text information
    for elem in root.iter():

        if elem.tag == 'TEXT':
            if elem.text:
                toks = elem.text.split()
                for i in range(len(toks)):
                    ## tokenize process
                    word_id = len(word_cps)
                    if verbose:
                        print(toks[i], words[word_id])
                    assert toks[i] == words[word_id]
                    tokenized = word_tokenize(toks[i])
                    for j in range(len(tokenized)):
                        tokc = Token(content=tokenized[j], tok_id=len(doc.tokens), sent_id=sent_ids[word_id])
                        doc.tokens.append(tokc)

                    word_cps.append(toks[i])

            for mention_elem in elem:
                toks = mention_elem.text.split()
                tok_ids = []
                for i in range(len(toks)):
                    ## tokenize process
                    word_id = len(word_cps)
                    if verbose:
                        print(toks[i], words[word_id])
                    assert toks[i] == words[word_id]

                    tokenized = word_tokenize(toks[i])
                    for j in range(len(tokenized)):
                        tok_ids.append(len(doc.tokens) + i)
                        tokc = Token(content=tokenized[j], tok_id=len(doc.tokens), sent_id=sent_ids[word_id])
                        doc.tokens.append(tokc)

                    word_cps.append(toks[i])
                if mention_elem.tag == "EVENT":
                    doc.addEvent(Event(content=' '.join(toks), tok_ids=tok_ids, **mention_elem.attrib))
                elif mention_elem.tag == "TIMEX3":
                    doc.addTimex(Timex(content=' '.join(toks), tok_ids=tok_ids, **mention_elem.attrib))
                elif mention_elem.tag == "SIGNAL":
                    doc.addSignal(Signal(content=' '.join(toks), tok_ids=tok_ids, **mention_elem.attrib))
                else:
                    raise Exception("Unknown mention type: %s" % (mention_elem.tag))
                if mention_elem.tail:
                    toks = mention_elem.tail.split()
                    for i in range(len(toks)):
                        ## tokenize process
                        word_id = len(word_cps)
                        if verbose:
                            print(toks[i], words[word_id])
                        assert toks[i] == words[word_id]

                        tokenized = word_tokenize(toks[i])
                        for j in range(len(tokenized)):
                            tokc = Token(content=tokenized[j], tok_id=len(doc.tokens), sent_id=sent_ids[word_id])
                            doc.tokens.append(tokc)

                        word_cps.append(toks[i])
    return doc


def load_timeml(file_name):
    root = ET.parse(file_name).getroot()
    doc = None

    ## read DOCID and DCT information
    for elem in root.iter():
        if elem.tag == "DOCID":
            doc = TimeMLDoc(docid=elem.text)
        if elem.tag == "DCT":
            for mention_elem in elem:
                doc.dct = Timex(content=mention_elem.text, **mention_elem.attrib)
                # doc.addTimex(Timex(content=mention_elem.text, **mention_elem.attrib))
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
            for mention_elem in elem:
                toks = mention_elem.text.split()
                tok_ids = [ len(doc.tokens) + i for i in range(len(toks))]
                for i in range(len(toks)):
                    tokc = Token(content=toks[i], tok_id=len(doc.tokens) + i)
                    doc.tokens.append(tokc)
                if mention_elem.tag == "EVENT":
                    doc.addEvent(Event(content=' '.join(toks), tok_ids=tok_ids, **mention_elem.attrib))
                elif mention_elem.tag == "TIMEX3":
                    doc.addTimex(Timex(content=' '.join(toks), tok_ids=tok_ids, **mention_elem.attrib))
                elif mention_elem.tag == "SIGNAL":
                    doc.addSignal(Signal(content=' '.join(toks), tok_ids=tok_ids, **mention_elem.attrib))
                else:
                    raise Exception("Unknown mention type: %s" % (mention_elem.tag))
                if mention_elem.tail:
                    toks = mention_elem.tail.split()
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
                temprel = TempLink(sour=doc.getMentionById(sour), targ=doc.getMentionById(targ), rel=rel)
                # words = word_tokenize(' '.join(doc.geneInterTokens(sour, targ)))
                # temprel.interwords = words
                # temprel.interpos = TempUtils.geneInterPostion(words)
                tokens = doc.geneSentTokens(sour, targ)
                temprel.interwords = ' '.join([ tok.content for tok in tokens ])
                print(temprel.interwords)
                temprel.interpos = TempUtils.geneSentPostion(tokens, sour.tok_ids[0], targ.tok_ids[-1])
                doc.addTlink(temprel)
            except Exception as ex:
                # print(doc_id, rel)
                # print(ex)
                pass

    return doc_list


def prepare_rels_from_anchorml(rel_file, timeml_dir):
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
                temprel = TempLink(sour=doc.getMentionById(sour), targ=doc.getMentionById(targ), rel=rel)
                # words = word_tokenize(' '.join(doc.geneInterTokens(sour, targ)))
                # temprel.interwords = words
                # temprel.interpos = TempUtils.geneInterPostion(words)
                tokens = doc.geneSentTokens(sour, targ)
                temprel.interwords = ' '.join([ tok.content for tok in tokens ])
                print(temprel.interwords)
                temprel.interpos = TempUtils.geneSentPostion(tokens, sour.tok_ids[0], targ.tok_ids[-1])
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

    def test_load_anchorml(self):
        doc = load_anchorml("/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL044_wsj_0173.tml")
#        print([ tok.content for tok in doc.tokens])
        event1 = doc.events[0]
        event2 = doc.events[1]
        print(event1.content, event1.tok_ids)
        print(event2.content, event2.tok_ids)
        print(doc.geneInterTokens(event1, event2))
        print([tok.content for tok in doc.tokens[:25]])
        print(doc.geneInterPostion(event1, event2))

    def test_load_anchorml2(self):
        doc = load_anchorml("/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL044_wsj_0173.tml")
        print(len(doc.tokens), doc.tokens[-1].tok_id)
        doc.setSentIds2mention()
        for key, timex in doc.timexs.items():
            print(key, timex.sent_id)

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

    def test_load_extraml(self):
        extraml_file = "/Users/fei-c/Resources/timex/original/timebank_1_2/data/extra/APW19980301.0720.tml"
        words, sent_ids = load_extraml(extraml_file)
        for w, s in zip(words[-50:], sent_ids[-50:]):
            print(w, s)

    def test_load_anchorml_sentid(self):
        anchor_file = "/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL001_APW19980301.0720.tml"
        anchor_file = "/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL002_APW19980306.1001.tml"
        # anchor_file = "/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL003_APW19980322.0749.tml"
        # anchor_file = "/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL004_APW19980501.0480.tml"
        # anchor_file = "/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL006_NYT19980424.0421.tml"
        # anchor_file = "/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL007_PRI19980303.2000.2550.tml" ## <turn>
        # anchor_file = "/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL008_SJMN91-06338157.tml" ## a mail format?
        # anchor_file = "/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL009_VOA19980303.1600.0917.tml"
        # anchor_file = "/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL013_VOA19980501.1800.0355.tml"

        filename = anchor_file.split('/')[-1].split('_')[-1]
        print(filename)
        extraml_file = "/Users/fei-c/Resources/timex/original/timebank_1_2/data/extra/%s" % (filename)
        doc = load_anchorml_sentid(anchor_file, extraml_file, verbose=1)
        for tok in doc.tokens:
            print(tok.tok_id, tok.sent_id, tok.content)

    def test_normalize_time(self):
        anchorml_dir = "/Users/fei-c/Resources/timex/Release0531/ALL"
        anchorml_list = [os.path.join(anchorml_dir, filename) for filename in sorted(os.listdir(anchorml_dir))]
        for filename in anchorml_list:
            try:
                doc = load_anchorml(filename)
            except Exception as ex:
                traceback.print_exc()
                print(filename, ex)
            print(len(doc.tokens), doc.tokens[-1].tok_id + 1)
            doc.setSentIds2mention()   # set a sent_id to each mention in a doc
            doc.dct.tanchor = normalize_time(doc.dct.value)
            print(doc.dct.tid, doc.dct.tanchor)
            print('-' * 80)
            for key, timex in doc.timexs.items():
                try:
                    # print(key, timex.content, timex.value, timex.mod, timex.anchorTimeID, timex.beginPoint, timex.endPoint)
                    timex.tanchor = normalize_time(timex.value)
                except Exception as ex:
                    traceback.print_exc()
                    print(key, timex.content, ex)
            print('-' * 80)
            for key, timex in doc.timexs.items():
                try:
                    if not timex.tanchor:
                        if timex.anchorTimeID:
                            timex.tanchor = normalize_relative(timex, doc.dct if timex.anchorTimeID == 't0' else doc.timexs[timex.anchorTimeID])
                        elif timex.endPoint:
                            timex.tanchor = normalize_relative(timex, doc.dct if timex.endPoint == 't0' else doc.timexs[timex.endPoint])
                        elif timex.beginPoint:
                            timex.tanchor = normalize_relative(timex, doc.dct if timex.beginPoint == 't0' else doc.timexs[timex.beginPoint])
                except Exception as ex:
                    traceback.print_exc()
                    print(key, timex.content, ex)
            for key, timex in doc.timexs.items():
                print(key, timex.content, timex.value, timex.tanchor)


    def test_generate_links(self):
        anchorml_dir = "/Users/fei-c/Resources/timex/Release0531/ALL"
        anchorml_list = [os.path.join(anchorml_dir, filename) for filename in sorted(os.listdir(anchorml_dir))]
        label_dic = {}
        non_count = 0
        for filename in anchorml_list:
            try:
                doc = load_anchorml(filename)
            except Exception as ex:
                traceback.print_exc()
                print(filename, ex)
            doc.setSentIds2mention()  # set a sent_id to each mention in a doc
            doc.normalize_timex_value()
            doc.normalize_event_value()
            for event in doc.events.values():
                if not event.tanchor:
                    non_count += 1
            # doc.geneEventDCTPair()
            # for link in doc.event_dct:
            #     label_dic[link.rel] = label_dic.setdefault(link.rel, 0) + 1
            doc.geneEventTimexPair(window=1)
            for link in doc.event_timex:
                label_dic[link.rel] = label_dic.setdefault(link.rel, 0) + 1

        all_count = sum([ value for value in label_dic.values()])
        print("number of links:", all_count, ", non event", non_count)
        for key in sorted(label_dic.keys()):
            value = label_dic[key]
            print("label %s, num %i, rate %.2f%%" % (key, value, value * 100 / all_count))

    def test_normalize_events(self):
        tb_dir = "/Users/fei-c/Resources/timex/Release0531/TimeBank/TML"
        tbd_dir = "/Users/fei-c/Resources/timex/Release0531/TimeBankDense/TML"
        anchorml_dir = tbd_dir
        anchorml_list = [os.path.join(anchorml_dir, filename) for filename in sorted(os.listdir(anchorml_dir))]
        for filename in anchorml_list:
            # if filename == "/Users/fei-c/Resources/timex/納品0521jsa/ALL/ALL066_wsj_0471.tml":
            try:
                doc = load_anchorml(filename)
            except Exception as ex:
                traceback.print_exc()
                print(filename, ex)
            doc.normalize_event_value(verbose=0)
