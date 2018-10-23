from TempNormalization import *
from TempOrder import InduceMethod
import TempUtils

# from nltk import sent_tokenize, word_tokenize
from TempSyntax import *


# corenlp = CoreNLPServer()
# sent_tokenize = corenlp.get_sent


class Token:

    def __init__(self, **args):
        self.content = args.setdefault('content', None)
        self.tok_id = args.setdefault('tok_id', None)
        self.sent_id = args.setdefault('sent_id', None)
        self.conll_id = args.setdefault('conll_id', None) ## token id for conll dependency format
        self.pos = args.setdefault('pos', None)

    @property
    def content(self):
        return self.__content

    @content.setter
    def content(self, content):
        self.__content = content

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, pos):
        self.__pos = pos

    @property
    def tok_id(self):
        return self.__tok_id

    @tok_id.setter
    def tok_id(self, tok_id):
        self.__tok_id = tok_id

    @property
    def sent_id(self):
        return self.__sent_id

    @sent_id.setter
    def sent_id(self, sent_id):
        self.__sent_id = sent_id

    @property
    def conll_id(self):
        return self.__conll_id

    @conll_id.setter
    def conll_id(self, conll_id):
        self.__conll_id = conll_id

    @property
    def mention_type(self):
        return type(self)


class Mention:
    def __init__(self, **args):
        self.content = args.setdefault('content', None)
        self.tok_ids = args.setdefault('tok_ids', None)
        self.sent_id = args.setdefault('sent_id', None)

    @property
    def content(self):
        return self.__content

    @content.setter
    def content(self, content):
        self.__content = content

    @property
    def tok_ids(self):
        return self.__tok_ids

    @tok_ids.setter
    def tok_ids(self, tok_ids):
        self.__tok_ids = tok_ids

    @property
    def sent_id(self):
        return self.__sent_id

    @sent_id.setter
    def sent_id(self, sent_id):
        self.__sent_id = sent_id
    
    @property
    def mention_type(self):
        return type(self)


class Signal(Mention):
    def __init__(self, **args):
        super().__init__(**args)
        self.sid = args.setdefault('sid', None)

    @property
    def id(self):
        return self.__sid

    @property
    def sid(self):
        return self.__sid

    @sid.setter
    def sid(self, sid):
        self.__sid = sid
        
    @property
    def mention_type(self):
        return "Signal"


class Timex(Mention):
    def __init__(self, **args):
        super().__init__(**args)
        self.tid = args.setdefault('tid', None)
        self.type = args.setdefault('type', None)
        self.value = args.setdefault('value', None)
        self.temporalFunction = args.setdefault('temporalFunction', None)
        self.functionInDocument = args.setdefault('functionInDocument', None)
        self.anchorTimeID = args.setdefault('anchorTimeID', None)
        self.beginPoint = args.setdefault('beginPoint', None)
        self.endPoint = args.setdefault('endPoint', None)
        self.mod = args.setdefault('mod', None)
        self.tanchor = args.setdefault('tanchor', None)

    @property
    def id(self):
        return self.__tid

    @property
    def tid(self):
        return self.__tid

    @tid.setter
    def tid(self, tid):
        self.__tid = tid

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, type):
        self.__type = type

    @property
    def mod(self):
        return self.__mod

    @mod.setter
    def mod(self, mod):
        self.__mod = mod

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        self.__value = value

    @property
    def tanchor(self):
        return self.__tanchor

    @tanchor.setter
    def tanchor(self, tanchor):
        self.__tanchor = tanchor

    @property
    def beginPoint(self):
        return self.__beginPoint

    @beginPoint.setter
    def beginPoint(self, beginPoint):
        self.__beginPoint = beginPoint

    @property
    def endPoint(self):
        return self.__endPoint

    @endPoint.setter
    def endPoint(self, endPoint):
        self.__endPoint = endPoint

    @property
    def anchorTimeID(self):
        return self.__anchorTimeID

    @anchorTimeID.setter
    def anchorTimeID(self, anchorTimeID):
        self.__anchorTimeID = anchorTimeID

    @property
    def temporalFunction(self):
        return self.__temporalFunction

    @temporalFunction.setter
    def temporalFunction(self, temporalFunction):
        self.__temporalFunction = temporalFunction

    @property
    def functionInDocument(self):
        return self.__functionInDocument

    @functionInDocument.setter
    def functionInDocument(self, functionInDocument):
        self.__functionInDocument = functionInDocument

    def isDCT(self):
        return True if self.__functionInDocument == "CREATION_TIME" else False

    @property
    def mention_type(self):
        return "DCT" if self.isDCT() else "Timex"


class EventBase(Mention):
    def __init__(self, **args):
        super().__init__(**args)
        self.eid = args.setdefault('eid', None)
        self.eclass = args.setdefault('class', None)
        self.value = args.setdefault('tanchor', None)
        self.tanchor = None
        self.daylen = 'Unknown'

    @property
    def eid(self):
        return self.__eid

    @eid.setter
    def eid(self, eid):
        self.__eid = eid

    @property
    def eclass(self):
        return self.__eclass

    @eclass.setter
    def eclass(self, eclass):
        self.__eclass = eclass

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        self.__value = value

    @property
    def daylen(self):
        return self.__daylen

    @daylen.setter
    def daylen(self, daylen):
        self.__daylen = daylen

    def normalize_value(self):
        if self.__value:
            self.__tanchor = normalize_tanchor(self.__value)

    def normalize_anchor(self):
        if self.__value:
            self.__tanchor = normalize_anchor(self.__value)

    @property
    def tanchor(self):
        return self.__tanchor

    @tanchor.setter
    def tanchor(self, tanchor):
        self.__tanchor = tanchor

class Event(EventBase):

    def __init__(self, **args):
        super().__init__(**args)
        self.eiid = args.setdefault('eiid', None)
        self.tense = args.setdefault('tense', None)
        self.aspect = args.setdefault('aspect', None)
        self.polarity = args.setdefault('polarity', None)
        self.pos = args.setdefault('pos', None)
        self.feat_inputs = {}  ## feat_inputs to be inputed into the model: {feat_name: feat_input}

    @property
    def id(self):
        return self.eid

    @property
    def eiid(self):
        return self.__eiid

    @eiid.setter
    def eiid(self, eiid):
        self.__eiid = eiid

    @property
    def tense(self):
        return self.__tense

    @tense.setter
    def tense(self, tense):
        self.__tense = tense

    @property
    def aspect(self):
        return self.__aspect

    @aspect.setter
    def aspect(self, aspect):
        self.__aspect = aspect

    @property
    def polarity(self):
        return self.__polarity

    @polarity.setter
    def polarity(self, polarity):
        self.__polarity = polarity

    @property
    def pos(self):
        return self.__pos

    @pos.setter
    def pos(self, pos):
        self.__pos = pos

    @property
    def mention_type(self):
        return "Event"


class TempLink:

    def __init__(self, **args):
        self.lid = args.setdefault('lid', None)
        self.sour = args.setdefault('sour', None)
        self.targ = args.setdefault('targ', None)
        self.rel = args.setdefault('rel', None)
        self.feat_inputs = {}  ## feat_inputs to be inputed into the model: {feat_name: feat_input}
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
    def features(self):
        return self.__features

    @features.setter
    def features(self, features):
        self.__features = features


    @property
    def link_type(self):
        if self.sour.mention_type != 'Event' and self.targ.mention_type == 'Event':
            return "%s-%s" % (self.targ.mention_type, self.sour.mention_type)
        else:
            return "%s-%s" % (self.sour.mention_type, self.targ.mention_type)


class TimeMLDoc:
    def __init__(self, **args):
        self.docid = args['docid']
        self.dct = None
        self.tokens = []
        self.events = {}
        self.timexs = {}
        self.signals = {}
        self.temp_links = {}
        self.syntaxer = TempSyntax()

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
        if not dct or dct.mention_type == 'DCT':
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
    def temp_links(self):
        return self.__temp_links

    @temp_links.setter
    def temp_links(self, temp_links):
        self.__temp_links = temp_links

    def addEvent(self, event):
        if event.mention_type == 'Event':
            self.events[event.eid] = event
        else:
            raise Exception('fail to add a non-Event object...%s' % (type(event)))

    def addTimex(self, timex):
        if timex.mention_type in ['Timex', 'DCT']:
            self.timexs[timex.tid] = timex
        else:
            raise Exception('fail to add a non-Timex object...%s' % (type(timex)))

    def addSignal(self, signal):
        if signal.mention_type == 'Signal':
            self.signals[signal.sid] = signal
        else:
            raise Exception('fail to add a non-Signal object...%s' % (signal.mention_type))

    def add_temp_link(self, link):
        if isinstance(link, TempLink):
            self.temp_links.setdefault(link.link_type, []).append(link)
        else:
            raise Exception('fail to add a non-TempLink object into event-dct...')

    def get_links_by_type(self, link_type):
        return self.temp_links.setdefault(link_type, [])

    def extendTokens(self, tokens):
        self.tokens.extend(tokens)

    ## assign sent_id and conll_id for each token in a doc
    def setSentIds2tok(self):

        corenlp = TempSyntax()
        sent_tokenize = corenlp.get_sent

        tok_seq = ' '.join([ tokc.content for tokc in self.tokens])
        sent_seq = sent_tokenize(tok_seq)

        index = 0
        for s_id in range(len(sent_seq)):
            sent = sent_seq[s_id]
            conll_id = 1
            while sent:
                if sent.startswith(self.tokens[index].content):
                    sent = sent[len(self.tokens[index].content):].strip()
                    self.tokens[index].sent_id = s_id + 1
                    self.tokens[index].conll_id = conll_id
                    conll_id += 1
                    index += 1
                else:
                    print(sent)
                    print(index, self.tokens[index].content)
                    raise Exception("Sentence and token aren't matching")


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

        begin_id = source.tok_ids[0]
        end_id = target.tok_ids[-1]

        for i in range(source.tok_ids[0], -1 , -1):
            if self.tokens[i].sent_id != source.sent_id:
                break
            begin_id = i

        for j in range(target.tok_ids[-1], len(self.tokens) , 1):
            if self.tokens[j].sent_id != target.sent_id:
                break
            end_id = j

        toks = self.tokens[begin_id: end_id + 1]
        return toks

    def geneSentOfMention(self, source):

        begin_id = source.tok_ids[0]
        end_id = source.tok_ids[-1]

        for i in range(source.tok_ids[0], -1 , -1):
            if self.tokens[i].sent_id != source.sent_id:
                break
            begin_id = i

        for j in range(source.tok_ids[-1], len(self.tokens) , 1):
            if self.tokens[j].sent_id != source.sent_id:
                break
            end_id = j

        toks = self.tokens[begin_id: end_id + 1]
        return toks

    def getSdpFromMentionToRoot(self, mention, direct='mention2root', dep_ver='SD'):
        sent_tokens = self.geneSentOfMention(mention)
        try:
            dep_graph = self.syntaxer.get_dep_graph(' '.join([ token.content.replace(' ', '-') for token in sent_tokens]), dep_ver=dep_ver)
            mention_conll_ids = [ self.tokens[tok_id].conll_id for tok_id in mention.tok_ids]
            if direct == 'mention2root':
                sdp_conll_ids = self.syntaxer.get_sdp(dep_graph, mention_conll_ids[0], 0)[:-1]
            elif direct == 'root2mention':
                sdp_conll_ids = self.syntaxer.get_sdp(dep_graph, 0, mention_conll_ids[0])[1:]
            else:
                raise Exception("[Error] Unknown sdp direct arg!!!")
            sdp_conll_ids = reformSDPforMention(mention_conll_ids, sdp_conll_ids)
            return sdp_conll_ids, mention_conll_ids, dep_graph
        except Exception as ex:
            print(ex)
            print(mention.content)
            print(' '.join([ tok.content for tok in sent_tokens]))
            return None

    def getSdpFeats(self, conll_ids, dep_graph):
        return ([ dep_graph.get_by_address(conll_id)['word'] for conll_id in conll_ids],
                [ dep_graph.get_by_address(conll_id)['tag'] for conll_id in conll_ids],
                [ dep_graph.get_by_address(conll_id)['rel'] for conll_id in conll_ids])

    def getTlinkListByMention(self, mention_id, link_types=['Event-DCT', 'Event-Timex']):
        tlinks = []
        for link_type in link_types:
            for link in self.temp_links[link_type]:
                if link.sour.id == mention_id or link.targ.id == mention_id:
                    tlinks.append(link)
        return tlinks


    def geneEventDCTPair(self, oper=False):
        lid = 0
        for eid, event in self.events.items():
            # print(InduceMethod.induce_relation(event, self.dct))
            link = TempLink(lid='led%i' % lid ,
                                        sour=event,
                                        targ=self.dct,
                                        rel=InduceMethod.induceRelationWithSourEvent(event, self.dct) if not oper else
                                            InduceMethod.induce_operation(event, self.dct))
            self.add_temp_link(link)
            lid += 1

    def geneEventTimexPair(self, sent_win, order='fixed', oper=False):
        lid = 0
        for eid, event in self.events.items():
            for tid, timex in self.timexs.items():
                if abs(timex.sent_id - event.sent_id) > sent_win:
                    continue

                if order == 'fixed':
                    sour, targ = event, timex
                else:
                    if event.tok_ids[0] <= timex.tok_ids[0]:  # specifying: a tlink is always from left to right
                        sour, targ = event, timex
                    else:
                        sour, targ = timex, event

                link = TempLink(lid='let%i' % lid ,
                                              sour=sour,
                                              targ=targ,
                                              rel=InduceMethod.induceRelationWithSourEvent(sour, targ) if not oper else
                                                  InduceMethod.induce_operation(sour, targ))
                tokens = self.geneSentTokens(sour, targ)
                link.interwords = [tok.content for tok in tokens]
                link.interpos = TempUtils.geneSentPostion(tokens, sour, targ)
                self.add_temp_link(link)
                lid += 1

    def geneEventsPair(self, sent_win, oper=False):
        lid = 0
        for sour_eid, sour_event in self.events.items():
            for targ_eid, targ_event in self.events.items():
                if 0 <= (targ_event.sent_id - sour_event.sent_id) <= sent_win:
                    if targ_event.tok_ids[0] > sour_event.tok_ids[0]:
                        link = TempLink(lid='lee%i' % lid ,
                                                      sour=sour_event,
                                                      targ=targ_event,
                                                      rel=InduceMethod.induce_relation(sour_event, targ_event)
                                                          if not oper else InduceMethod.induce_operation(sour_event, targ_event))
                        tokens = self.geneSentTokens(sour_event, targ_event)
                        link.interwords = [tok.content for tok in tokens]
                        link.interpos = TempUtils.geneSentPostion(tokens, sour_event, targ_event)
                        self.add_temp_link(link)

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

    def normalize_event_value(self, anchor_type='anchor0', verbose=0):
        for key, event in self.events.items():
            try:
                if anchor_type == 'anchor0':
                    event.tanchor = normalize_anchor(event.value)
                elif anchor_type == 'anchor1':
                    event.tanchor = normalize_tanchor(event.value)
                event.daylen = dayLengthOfMention(event.tanchor)
                # print(event.value, normalize_anchor(event.value))
            except Exception as ex:
                print("Normalize event error:", key, event.value)

import unittest

class TestEventMention(unittest.TestCase):

    def test_event(self):
        e = Event(content='capture', pos='NN', eid=1, eiid=2)
        print(e.content, e.pos, e.eid, e.eiid, e.tanchor, e.mention_type)
        if isinstance(e, Mention):
            print(type(e))

    def test_timex(self):
        t = Timex(content='next week', tid=1, value='1998-01-01')
        print(t.content, t.value, t.tid, t.mention_type)

class TestTempLink(unittest.TestCase):

    def test_templink(self):
        e = Event(content='capture', pos='NN', eid=1, eiid=2)
        t = Timex(content='next week', tid=1, value='1998-01-01')
        tlink = TempLink(lid=1, sour=e, targ=e, relType="before")
        print(tlink.lid, tlink.link_type)

    def test_doc(self):
        doc = TimeMLDoc(docid='ABC19980108.1830.0711')
        e = Event(content='capture', pos='NN', eid=1, eiid=2)
        t = Timex(content='next week', tid=1, value='1998-01-01', functionInDocument="CREATION_TIME")
        doc.addEvent(e)
        doc.dct = t
        print(doc.docid, doc.events, doc.timexs, doc.dct)