from TempMention import Mention,Event,Timex
from nltk import sent_tokenize, word_tokenize

from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta

class TimeMLDoc:
    def __init__(self, **args):
        self.docid = args['docid']
        self.dct = None
        self.tokens = []
        self.events = {}
        self.timexs = {}
        self.signals = {}
        self.tlinks = []

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
        if dct == None or dct.category == 'DCT':
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
    def tlinks(self):
        return self.__tlinks

    @tlinks.setter
    def tlinks(self, tlinks):
        self.__tlinks = tlinks

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

    def addTlink(self, link):
        if isinstance(link, TempLink):
            self.tlinks.append(link)
        else:
            raise Exception('fail to add a non-TempLink object...')

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

    def geneMentionPair(self, window=1):
        lid = 0
        for eid, event in self.events.items():
            for tid, timex in self.timexs.items():
                if event.tok_ids[0] <= timex.tok_ids[0]:  # specifying: a tlink is always from left to right
                    sour, targ = event, timex
                else:
                    sour, targ = timex, event
                self.addTlink(TempLink(lid='l%i' % lid , sour=sour, targ=targ, rel=None) )


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


def last_day_of_month(date):
    if date.month == 12:
        return date.replace(day=31)
    return date.replace(month=date.month+1, day=1) - timedelta(days=1)

def regular_season(value):
    year = value.split('-')[0]
    season = value.split('-')[1]
    if season in ['SP']:
        begin = datetime.strptime("%s-03-21" % year, '%Y-%m-%d')
    elif season in ['SU']:
        begin = datetime.strptime("%s-06-21" % year, '%Y-%m-%d')
    elif season in ['F', 'FA']:
        begin = datetime.strptime("%s-09-21" % year, '%Y-%m-%d')
    elif season in ['W', 'WI']:
        begin = datetime.strptime("%s-12-21" % year, '%Y-%m-%d')
    end = begin + relativedelta(months=3) - timedelta(days=1)
    return (begin, begin, end, end)

def regular_quarter(value):
    year = value.split('-')[0]
    quarter = int(value.split('-')[1][1])
    if quarter == 1:
        begin = datetime.strptime("%s-01-01" % year, '%Y-%m-%d')
    elif quarter == 2:
        begin = datetime.strptime("%s-04-01" % year, '%Y-%m-%d')
    elif quarter == 3:
        begin = datetime.strptime("%s-07-01" % year, '%Y-%m-%d')
    elif quarter == 4:
        begin = datetime.strptime("%s-10-01" % year, '%Y-%m-%d')
    end = begin + relativedelta(months=3) - timedelta(days=1)
    return (begin, begin, end, end)


def normalize_time(value):
    value = value.strip().split('T')[0]
    if value[0].isdigit():
        if value.count('-') == 2 and value.split('-')[1].isdigit():
            return (datetime.strptime(value, '%Y-%m-%d'), datetime.strptime(value, '%Y-%m-%d'))
        elif value.count('-') == 2 and value.split('-')[1][0] == 'W' and value.split('-')[2] == "WE":
            year = value.split('-')[0]
            week = int(value.split('-')[1][1:]) - 1
            begin  = datetime.strptime("%s-W%i-0" % (year, week), "%Y-W%W-%w") + timedelta(days=6)
            end = begin + timedelta(days=1)
            return (begin, begin, end, end)
        elif value.count('-') == 1:
            if value.split('-')[-1][0].isdigit():
                begin = datetime.strptime(value, '%Y-%m')
                return (begin, begin, last_day_of_month(begin), last_day_of_month(begin))
            elif value.split('-')[-1][0] in ['W', 'w'] and value.split('-')[-1][1:].isdigit():
                year = value.split('-')[0]
                week = int(value.split('-')[1][1:]) - 1
                begin = datetime.strptime("%s-W%i-0" % (year, week), "%Y-W%W-%w")
                end = begin + timedelta(days=6)
                return (begin, begin, end, end)
            elif value.split('-')[-1][0] in ['Q', 'q']:
                return regular_quarter(value)
            elif value.split('-')[-1] in ['SP', 'SU', 'F', 'FA', 'W', 'WI']:
                return regular_season(value)
            else:
                print("time value:", value)
        elif value.count('-') == 0:
            begin = datetime.strptime(value, '%Y')
            end = begin + relativedelta(months=12) - timedelta(days=1)
            return (begin, begin, end, end)
    else:
        return None

def normalize_relative(timex, relative_timex):
    if not relative_timex.tanchor:
        return None
    else:
        if timex.value[0] == 'P' and timex.value[1:-1].isdigit() and timex.value[-1] in ['Y', 'M', 'W', 'D']:
            if timex.endPoint:
                end = relative_timex.tanchor[-1]
                if timex.value[-1] == 'Y':
                    begin = end - relativedelta(years=int(timex.value[1:-1])) + relativedelta(days=1)
                elif timex.value[-1] == 'M':
                    begin = end - relativedelta(months=int(timex.value[1:-1])) + relativedelta(days=1)
                elif timex.value[-1] == 'W':
                    begin = end - relativedelta(weeks=int(timex.value[1:-1])) + relativedelta(days=1)
                elif timex.value[-1] == 'D':
                    begin = end - relativedelta(days=int(timex.value[1:-1])) + relativedelta(days=1)
                return (begin, begin, end, end)
            elif timex.beginPoint:
                begin = relative_timex.tanchor[0]
                if timex.value[-1] == 'Y':
                    end = begin + relativedelta(years=int(timex.value[1:-1])) - relativedelta(days=1)
                elif timex.value[-1] == 'M':
                    end = begin + relativedelta(months=int(timex.value[1:-1])) - relativedelta(days=1)
                elif timex.value[-1] == 'W':
                    end = begin + relativedelta(weeks=int(timex.value[1:-1])) - relativedelta(days=1)
                elif timex.value[-1] == 'D':
                    end = begin + relativedelta(days=int(timex.value[1:-1])) - relativedelta(days=1)
                return (begin, begin, end, end)
        elif timex.value == "PAST_REF":
            return (None, relative_timex.tanchor[-1])
        elif timex.value == "FUTURE_REF":
            return (relative_timex.tanchor[0], None)
        elif timex.value == "PRESENT_REF":
            return (relative_timex.tanchor[0], relative_timex.tanchor[1])
        else:
            return None






class RelationGenerator():

    ## we transfer an tanchor into 2 formats
    ## (after YYYY-MM-DD, before YYYY-MM-DD)  one day (certain and uncertain)
    ## (YYYY-MM-DD, YYYY-MM-DD, YYYY-MM-DD, YYYY-MM-DD) multiple days
    def __init__(self):
        self.label_dic = {
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

    def compare_2single(self, sour, targ):

        def certain_certain(sour, targ):
            if sour[0] < targ[0]:
                return "before"
            elif sour[0] > targ[0]:
                return "after"
            elif sour[0] == targ[0]:
                return "same"

        def certain_uncertain(sour, targ):
            if sour[0] <= targ[0]:
                return "before"
            elif sour[0] >= targ[1]:
                return "after"
            else:
                return "vague"

        def uncertain_uncertain(sour, targ):
            if sour[1] <= targ[0]:
                return "before"
            elif sour[0] >= targ[1]:
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
            return self.label_dic[certain_uncertain(targ, sour)]
        else:                                                # uncertain - uncertain single-days
            return uncertain_uncertain(sour, targ)

    def compare_singlemultiple(self, sour, targ):
        targ_begin = (targ[0], targ[1])
        targ_end = (targ[2], targ[3])

        if self.compare_2single(sour, targ_begin) in ["before"]:
            return "before"
        elif self.compare_2single(sour, targ_end) in ["after"]:
            return "after"
        elif self.compare_2single(sour, targ_begin) in ["after"] and self.compare_2single(sour, targ_end) in ["before"]:
            return "is_included"
        elif self.compare_2single(sour, targ_begin) in ["same"]:
            return "begin"
        elif self.compare_2single(sour, targ_end) in ["same"]:
            return "end"
        else:
            return "vague"

    def compare_2multiple(self, sour, targ):
        sour_begin, sour_end = (sour[0], sour[1]), (sour[2], sour[3])
        targ_begin, targ_end = (targ[0], targ[1]), (targ[2], targ[3])

        if self.compare_2single(sour_end, targ_begin) in ["before"]:
            return "before"
        elif self.compare_2single(sour_begin, targ_end) in ["after"]:
            return "after"
        elif self.compare_2single(sour_begin, targ_begin) in ["same"] and self.compare_2single(sour_end, targ_end) in ["same"]:
            return "samespan"
        elif self.compare_2single(sour_begin, targ_begin) in ["after"] and self.compare_2single(sour_end, targ_end) in ["before"]:
            return "is_included"
        elif self.compare_2single(sour_begin, targ_begin) in ["before"] and self.compare_2single(sour_end, targ_end) in ["after"]:
            return "include"
        elif self.compare_2single(sour_begin, targ_begin) in ["same"] and self.compare_2single(sour_end, targ_end) in ["after"]:
            return "begun_by"
        elif self.compare_2single(sour_begin, targ_begin) in ["same"] and self.compare_2single(sour_end, targ_end) in ["before"]:
            return "begin"
        elif self.compare_2single(sour_begin, targ_begin) in ["after"] and self.compare_2single(sour_end, targ_end) in ["same"]:
            return "end"
        elif self.compare_2single(sour_begin, targ_begin) in ["before"] and self.compare_2single(sour_end, targ_end) in ["same"]:
            return "ended_by"
        elif self.compare_2single(sour_begin, targ_begin) in ["before"] and self.compare_2single(sour_end, targ_end) in ["before"] \
                and self.compare_2single(sour_end, targ_begin) in ["after", "same"]:
            return "overlap"
        elif self.compare_2single(sour_begin, targ_begin) in ["after"] and self.compare_2single(sour_end, targ_end) in ["after"] \
                and self.compare_2single(sour_begin, targ_end) in ["before", "same"]:
            return "overlap"
        elif self.compare_2single(sour_begin, targ_begin) in ["partialvague"] and self.compare_2single(sour_end, targ_end) in ["same"]:
            return "partialvague"
        elif self.compare_2single(sour_begin, targ_begin) in ["same"] and self.compare_2single(sour_end, targ_end) in ["partialvague"]:
            return "partialvague"
        elif self.compare_2single(sour_begin, targ_begin) in ["partialvague"] and self.compare_2single(sour_end, targ_end) in ["partialvague"]:
            return "partialvague"
        else:
            return "vague"

import unittest

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
