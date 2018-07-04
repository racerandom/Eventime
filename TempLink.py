import unittest
import re

from TempMention import Mention,Event,Timex
from TempNormalization import *
import TempUtils

from nltk import sent_tokenize, word_tokenize


class TimeMLDoc:
    def __init__(self, **args):
        self.docid = args['docid']
        self.dct = None
        self.tokens = []
        self.events = {}
        self.timexs = {}
        self.signals = {}
        self.event_dct = []
        self.event_timex = []
        self.event_event = []

    @property
    def docid(self):
        return self.__docid

    @docid.setter
    def docid(self, docid):
        self.__docid = docid

    @property
    def dct(self):
        return self.__dct

    @dct.setter
    def dct(self, dct):
        if not dct or dct.category == 'DCT':
            self.__dct = dct
        else:
            raise Exception('TimeMLDoc cannot assign a non-DCT object as dct...')

    @property
    def events(self):
        return self.__events

    @events.setter
    def events(self, events):
        self.__events = events

    @property
    def tokens(self):
        return self.__tokens

    @tokens.setter
    def tokens(self, tokens):
        self.__tokens = tokens

    @property
    def timexs(self):
        return self.__timexs

    @timexs.setter
    def timexs(self, timexs):
        self.__timexs = timexs

    @property
    def signals(self):
        return self.__signals

    @signals.setter
    def signals(self, signals):
        self.__signals = signals

    @property
    def event_dct(self):
        return self.__event_dct

    @event_dct.setter
    def event_dct(self, event_dct):
        self.__event_dct = event_dct

    @property
    def event_timex(self):
        return self.__event_timex

    @event_timex.setter
    def event_timex(self, event_timex):
        self.__event_timex = event_timex

    @property
    def event_event(self):
        return self.__event_event

    @event_event.setter
    def event_event(self, event_event):
        self.__event_event = event_event

    def addEvent(self, event):
        if event.category == 'Event':
            self.events[event.eid] = event
        else:
            raise Exception('fail to add a non-Event object...%s' % (type(event)))

    def addTimex(self, timex):
        if timex.category in ['Timex', 'DCT']:
            self.timexs[timex.tid] = timex
        else:
            raise Exception('fail to add a non-Timex object...%s' % (type(timex)))

    def addSignal(self, signal):
        if signal.category == 'Signal':
            self.signals[signal.sid] = signal
        else:
            raise Exception('fail to add a non-Signal object...%s' % (signal.category))

    def add_event_dct(self, link):
        if isinstance(link, TempLink):
            self.event_dct.append(link)
        else:
            raise Exception('fail to add a non-TempLink object into event-dct...')

    def add_event_timex(self, link):
        if isinstance(link, TempLink):
            self.event_timex.append(link)
        else:
            raise Exception('fail to add a non-TempLink object into event-timex...')

    def add_event_event(self, link):
        if isinstance(link, TempLink):
            self.event_event.append(link)
        else:
            raise Exception('fail to add a non-TempLink object into event-event...')

    def extendTokens(self, tokens):
        self.tokens.extend(tokens)

    def setSentIds2tok(self):
        tok_seq = ' '.join([ tokc.content for tokc in self.tokens])
        sent_seq = sent_tokenize(tok_seq)
        index = 0
        for s_id in range(len(sent_seq)):
            for w_id in range(len(sent_seq[s_id].split())):
                # print(s_id, sent_seq[s_id], index, self.tokens[index].content)
                self.tokens[index].sent_id = s_id + 1
                index += 1

    def setSentIds2mention(self):
        self.setSentIds2tok()
        for event in self.events.values():
            event.sent_id = self.tokens[event.tok_ids[0]].sent_id
        for timex in self.timexs.values():
            timex.sent_id = self.tokens[timex.tok_ids[0]].sent_id
        for signal in self.signals.values():
            signal.sent_id = self.tokens[signal.tok_ids[0]].sent_id

    def getMentionById(self, mention_id):
        if mention_id[0] == 'e':
            return self.events[mention_id]
        elif mention_id[0] == 't':
            return self.timexs[mention_id]
        elif mention_id[0] == 's':
            return self.signals[mention_id]
        else:
            raise Exception('[ERROR]: Unrecognized Mention Id...')

    def geneInterTokens(self, source_id, target_id):
        source = self.getMentionById(source_id)
        target = self.getMentionById(target_id)
        left_id = source.tok_ids[0]
        right_id = target.tok_ids[-1]
        toks = self.tokens[left_id: right_id + 1]
        return [ tok.content for tok in toks]

    def geneSentTokens(self, source, target):
        # source = self.getMentionById(source_id)
        # target = self.getMentionById(target_id)

        left_id = source.tok_ids[0]
        right_id = target.tok_ids[-1]

        for i in range(source.tok_ids[0], -1 , -1):
            if self.tokens[i].sent_id != source.sent_id:
                break
            left_id = i

        for j in range(target.tok_ids[-1], len(self.tokens) , 1):
            if self.tokens[j].sent_id != target.sent_id:
                break
            right_id = j

        toks = self.tokens[left_id: right_id + 1]
        return toks

    def geneEventDCTPair(self, window=1):
        lid = 0
        for eid, event in self.events.items():
            # print(InduceMethod.induce_relation(event, self.dct))
            link = TempLink(lid='led%i' % lid ,
                                        sour=event,
                                        targ=self.dct,
                                        rel=InduceMethod.induce_relation(event, self.dct))
            self.add_event_dct(link)
            lid += 1

    def geneEventTimexPair(self, sent_win):
        lid = 0
        for eid, event in self.events.items():
            for tid, timex in self.timexs.items():
                if abs(timex.sent_id - event.sent_id) > sent_win:
                    continue
                if event.tok_ids[0] <= timex.tok_ids[0]:  # specifying: a tlink is always from left to right
                    sour, targ = event, timex
                else:
                    sour, targ = timex, event
                link = TempLink(lid='let%i' % lid ,
                                              sour=sour,
                                              targ=targ,
                                              rel=InduceMethod.induce_relation(sour, targ))
                tokens = self.geneSentTokens(sour, targ)
                link.interwords = [tok.content for tok in tokens]
                link.interpos = TempUtils.geneSentPostion(tokens, sour.tok_ids[-1], targ.tok_ids[-1])
                self.add_event_timex(link)
                lid += 1

    def geneEventsPair(self, sent_win):
        lid = 0
        for sour_eid, sour_event in self.events.items():
            for targ_eid, targ_event in self.events.items():
                if 0 <= (targ_event.sent_id - sour_event.sent_id) <= sent_win:
                    if targ_event.tok_ids[0] > sour_event.tok_ids[0]:
                        link = TempLink(lid='led%i' % lid ,
                                                      sour=sour_event,
                                                      targ=targ_event,
                                                      rel=InduceMethod.induce_relation(sour_event, targ_event))
                        tokens = self.geneSentTokens(sour_event, targ_event)
                        link.interwords = [tok.content for tok in tokens]
                        link.interpos = TempUtils.geneSentPostion(tokens, sour_event.tok_ids[-1], targ_event.tok_ids[-1])
                        self.add_event_event(link)

    def normalize_timex_value(self, verbose=0):
        self.dct.tanchor = normalize_time(self.dct.value)
        for key, timex in self.timexs.items():
            if verbose:
                print(key, timex.content, timex.value, timex.mod, timex.anchorTimeID, timex.beginPoint, timex.endPoint)
            timex.tanchor = normalize_time(timex.value)
        for key, timex in self.timexs.items():
            try:
                if not timex.tanchor:
                    if timex.anchorTimeID:
                        timex.tanchor = normalize_relative(timex, self.dct if timex.anchorTimeID == 't0' else self.timexs[
                            timex.anchorTimeID])
                    elif timex.endPoint:
                        timex.tanchor = normalize_relative(timex, self.dct if timex.endPoint == 't0' else self.timexs[
                            timex.endPoint])
                    elif timex.beginPoint:
                        timex.tanchor = normalize_relative(timex, self.dct if timex.beginPoint == 't0' else self.timexs[
                            timex.beginPoint])
            except Exception as ex:
                print("Normalize timex error:", key, timex.value)
        if verbose:
            for key, timex in self.timexs.items():
                print(key, timex.content, timex.value, timex.tanchor)

    def normalize_event_value(self, verbose=0):
        for key, event in self.events.items():
            try:
                event.normalize_value()
            except Exception as ex:
                print("Normalize event error:", key, event.value)


class TempLink:

    def __init__(self, **args):
        self.lid = args.setdefault('lid', None)
        self.sour = args.setdefault('sour', None)
        self.targ = args.setdefault('targ', None)
        self.rel = args.setdefault('rel', None)
        self.check()

    def check(self):
        if not isinstance(self.sour, Mention) or not isinstance(self.targ, Mention):
            raise Exception("The source or target is not a Mention class")

    @property
    def lid(self):
        return self.__lid

    @lid.setter
    def lid(self, lid):
        self.__lid = lid

    @property
    def sour(self):
        return self.__sour

    @sour.setter
    def sour(self, sour):
        self.__sour = sour

    @property
    def targ(self):
        return self.__targ

    @targ.setter
    def targ(self, targ):
        self.__targ = targ

    @property
    def rel(self):
        return self.__rel

    @rel.setter
    def rel(self, rel):
        self.__rel = rel

    @property
    def interwords(self):
        return self.__interwords

    @interwords.setter
    def interwords(self, interwords):
        self.__interwords = interwords

    @property
    def interpos(self):
        return self.__interpos

    @interpos.setter
    def interpos(self, interpos):
        self.__interpos = interpos

    @property
    def category(self):
        return "%s-%s" % (self.sour.category, self.targ.category)


class InduceMethod():

    ## we transfer an tanchor into 2 formats
    ## (after YYYY-MM-DD, before YYYY-MM-DD)  one day (certain and uncertain)
    ## (YYYY-MM-DD, YYYY-MM-DD, YYYY-MM-DD, YYYY-MM-DD) multiple days
    label_dic = {
            "after": "before",
            "before": "after",
            "include": "is_included",
            "is_included": "include",
            "same": "same",
            "samespan": "samespan",
            "vague": "vague",
            "partialvague": "partialvague",
            "overlap": "overlap",
            "begin": "begun_by",
            "begun_by": "begin",
            "end": "ended_by",
            "ended_by": "end",
            }

    @staticmethod
    def reverse_relation(rel):
        return InduceMethod.label_dic[rel]

    @staticmethod
    def compare_2single(sour, targ):

        def certain_certain(sour, targ):
            if sour[0] < targ[0]:
                return "before"
            elif sour[0] > targ[0]:
                return "after"
            elif sour[0] == targ[0]:
                return "same"

        def certain_uncertain(sour, targ):
            if None not in [sour[0], targ[0]] and sour[0] <= targ[0]:
                return "before"
            elif None not in [sour[0], targ[1]] and sour[0] >= targ[1]:
                return "after"
            else:
                return "vague"

        def uncertain_uncertain(sour, targ):
            if None not in [sour[1], targ[0]] and sour[1] <= targ[0]:
                return "before"
            elif None not in [sour[0], targ[1]] and sour[0] >= targ[1]:
                return "after"
            elif sour[0] == targ[0] and sour[1] == targ[1]:
                return "partialvague"
            else:
                return "vague"

        if not sour or not targ:
            return "vague"
        elif sour[0] == sour[1] and targ[0] == targ[1]:    # two certain single-days
            return certain_certain(sour, targ)
        elif sour[0] == sour[1] and targ[0] != targ[1]:     # certain - uncertain single-days
            return certain_uncertain(sour, targ)
        elif sour[0] != sour[1] and targ[0] == targ[1]:     # uncertain - certain single-days
            return InduceMethod.reverse_relation(certain_uncertain(targ, sour))
        else:                                                # uncertain - uncertain single-days
            return uncertain_uncertain(sour, targ)

    @staticmethod
    def compare_singlemultiple(sour, targ):
        targ_begin = (targ[0], targ[1])
        targ_end = (targ[2], targ[3])

        if InduceMethod.compare_2single(sour, targ_begin) in ["before"]:
            return "before"
        elif InduceMethod.compare_2single(sour, targ_end) in ["after"]:
            return "after"
        elif InduceMethod.compare_2single(sour, targ_begin) in ["after"] and InduceMethod.compare_2single(sour, targ_end) in ["before"]:
            return "is_included"
        elif InduceMethod.compare_2single(sour, targ_begin) in ["same"]:
            return "begin"
        elif InduceMethod.compare_2single(sour, targ_end) in ["same"]:
            return "end"
        else:
            return "vague"

    @staticmethod
    def compare_2multiple(sour, targ):
        sour_begin, sour_end = (sour[0], sour[1]), (sour[2], sour[3])
        targ_begin, targ_end = (targ[0], targ[1]), (targ[2], targ[3])

        if InduceMethod.compare_2single(sour_end, targ_begin) in ["before"]:
            return "before"
        elif InduceMethod.compare_2single(sour_begin, targ_end) in ["after"]:
            return "after"
        elif InduceMethod.compare_2single(sour_begin, targ_begin) in ["same"] and InduceMethod.compare_2single(sour_end, targ_end) in ["same"]:
            return "samespan"
        elif InduceMethod.compare_2single(sour_begin, targ_begin) in ["after"] and InduceMethod.compare_2single(sour_end, targ_end) in ["before"]:
            return "is_included"
        elif InduceMethod.compare_2single(sour_begin, targ_begin) in ["before"] and InduceMethod.compare_2single(sour_end, targ_end) in ["after"]:
            return "include"
        elif InduceMethod.compare_2single(sour_begin, targ_begin) in ["same"] and InduceMethod.compare_2single(sour_end, targ_end) in ["after"]:
            return "begun_by"
        elif InduceMethod.compare_2single(sour_begin, targ_begin) in ["same"] and InduceMethod.compare_2single(sour_end, targ_end) in ["before"]:
            return "begin"
        elif InduceMethod.compare_2single(sour_begin, targ_begin) in ["after"] and InduceMethod.compare_2single(sour_end, targ_end) in ["same"]:
            return "end"
        elif InduceMethod.compare_2single(sour_begin, targ_begin) in ["before"] and InduceMethod.compare_2single(sour_end, targ_end) in ["same"]:
            return "ended_by"
        elif InduceMethod.compare_2single(sour_begin, targ_begin) in ["before"] and InduceMethod.compare_2single(sour_end, targ_end) in ["before"] \
                and InduceMethod.compare_2single(sour_end, targ_begin) in ["after", "same"]:
            return "overlap"
        elif InduceMethod.compare_2single(sour_begin, targ_begin) in ["after"] and InduceMethod.compare_2single(sour_end, targ_end) in ["after"] \
                and InduceMethod.compare_2single(sour_begin, targ_end) in ["before", "same"]:
            return "overlap"
        elif InduceMethod.compare_2single(sour_begin, targ_begin) in ["partialvague"] and InduceMethod.compare_2single(sour_end, targ_end) in ["same"]:
            return "partialvague"
        elif InduceMethod.compare_2single(sour_begin, targ_begin) in ["same"] and InduceMethod.compare_2single(sour_end, targ_end) in ["partialvague"]:
            return "partialvague"
        elif InduceMethod.compare_2single(sour_begin, targ_begin) in ["partialvague"] and InduceMethod.compare_2single(sour_end, targ_end) in ["partialvague"]:
            return "partialvague"
        else:
            return "vague"

    @staticmethod
    def induce_relation(sour, targ):
        if not sour.tanchor or not targ.tanchor:
            return "vague"
        elif len(sour.tanchor) == 2 and len(targ.tanchor) == 2:
            return InduceMethod.compare_2single(sour.tanchor, targ.tanchor)
        elif len(sour.tanchor) == 2 and len(targ.tanchor) == 4:
            return InduceMethod.compare_singlemultiple(sour.tanchor, targ.tanchor)
        elif len(sour.tanchor) == 4 and len(targ.tanchor) == 2:
            return InduceMethod.reverse_relation(InduceMethod.compare_singlemultiple(targ.tanchor, sour.tanchor))
        elif len(sour.tanchor) == 4 and len(targ.tanchor) == 4:
            return InduceMethod.compare_2multiple(sour.tanchor, targ.tanchor)


class TestTempLink(unittest.TestCase):

    def test_templink(self):
        e = Event(content='capture', pos='NN', eid=1, eiid=2)
        t = Timex(content='next week', tid=1, value='1998-01-01')
        tlink = TempLink(lid=1, sour=e, targ=e, relType="before")
        print(tlink.lid, tlink.category)

    def test_doc(self):
        doc = TimeMLDoc(docid='ABC19980108.1830.0711')
        e = Event(content='capture', pos='NN', eid=1, eiid=2)
        t = Timex(content='next week', tid=1, value='1998-01-01', functionInDocument="CREATION_TIME")
        doc.addEvent(e)
        doc.dct = t
        print(doc.docid, doc.events, doc.timexs, doc.dct)
