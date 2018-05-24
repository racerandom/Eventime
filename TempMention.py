class Token:
    def __init__(self, **args):
        self.content = args.setdefault('content', None)
        self.tok_id = args.setdefault('tok_id', None)
        self.sent_id = args.setdefault('sent_id', None)

    @property
    def content(self):
        return self.__content

    @content.setter
    def content(self, content):
        self.__content = content

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
    def category(self):
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
    def category(self):
        return type(self)

class Signal(Mention):
    def __init__(self, **args):
        super().__init__(**args)
        self.sid = args.setdefault('sid', None)

    @property
    def sid(self):
        return self.__sid

    @sid.setter
    def sid(self, sid):
        self.__sid = sid
        
    @property
    def category(self):
        return "Signal"

class Timex(Mention):
    def __init__(self, **args):
        super().__init__(**args)
        self.tid = args.setdefault('tid', None)
        self.type = args.setdefault('type', None)
        self.value = args.setdefault('value', None)
        self.temporalFunction = args.setdefault('temporalFunction', None)
        self.functionInDocument = args.setdefault('functionInDocument', None)

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
    def type(self, types):
        self.__type = type

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        self.__value = value

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
    def category(self):
        return "DCT" if self.isDCT() else "Timex"

class EventBase(Mention):
    def __init__(self, **args):
        super().__init__(**args)
        self.eid = args.setdefault('eid', None)
        self.eclass = args.setdefault('class', None)
        self.tanchor = args.setdefault('tanchor', None)

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
    def category(self):
        return "Event"

import unittest

class TestEventMention(unittest.TestCase):

    def test_event(self):
        e = Event(content='capture', pos='NN', eid=1, eiid=2)
        print(e.content, e.pos, e.eid, e.eiid, e.tanchor, e.category)
        if isinstance(e, Mention):
            print(type(e))

    def test_timex(self):
        t = Timex(content='next week', tid=1, value='1998-01-01')
        print(t.content, t.value, t.tid, t.category)