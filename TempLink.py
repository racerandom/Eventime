from TempMention import Mention,Event,Timex

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
