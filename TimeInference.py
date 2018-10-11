# coding=utf-8

import os
from TimeMLReader import *
from TempData import *


def day_length(time):
    if not time:
        return None
    elif len(time) == 2:
        return 'single'
    elif len(time) == 4:
        return 'multiple'
    else:
        return None

def updateBeginAfter(event_time, date):
    if not event_time[0] or date > event_time[0]:
        event_time[0] = date
    return event_time

def updateBeginBefore(event_time, date):
    if not event_time[1] or date < event_time[1]:
        event_time[1] = date
    return event_time

def updateEndAfter(event_time, date):
    if not event_time[2] or date > event_time[2]:
        event_time[2] = date
    return event_time

def updateEndBefore(event_time, date):
    if not event_time[3] or date < event_time[3]:
        event_time[3] = date
    return event_time


class TimeInference():

    @staticmethod
    def inferTimeByOrder(event_time, time, order):
        if time:
            if order in ['same']:
                if day_length(time) == 'single':
                    event_time[0], event_time[1] = time[0], time[1]
                elif day_length(time) == 'multiple':
                    event_time = list(time)
            elif order in ['after_single']:
                slot = time[1] if day_length(time) == 'single' else time[3]
                updateBeginAfter(event_time, slot)
            elif order in ['after_multiple']:
                slot = time[1] if day_length(time) == 'single' else time[3]
                updateBeginAfter(event_time, slot)
                updateEndAfter(event_time, slot)
            elif order in ['before_single']:
                updateBeginBefore(event_time, time[0])
            elif order in ['before_multiple']:
                updateBeginBefore(event_time, time[0])
                updateEndBefore(event_time, time[0])
            elif order in ['include']:
                updateBeginBefore(event_time, time[0])
                slot = time[1] if day_length(time) == 'single' else time[3]
                updateEndAfter(event_time, slot)
            elif order in ['is_included_single']:
                updateBeginAfter(event_time, time[0])
                updateBeginBefore(event_time, time[3])
            elif order in ['is_included_multiple']:
                updateBeginAfter(event_time, time[0])
                updateBeginBefore(event_time, time[3])
                updateEndBefore(event_time, time[3])
            elif order in ['begun_by']:
                event_time[0], event_time[1] = time[0], time[0]
                updateEndAfter(event_time, time[0])
            elif order in ['ended_by']:
                event_time[2], event_time[3] = time[1], time[1]
                updateBeginBefore(event_time, time[1])
            elif order in ['begin']:
                event_time[0], event_time[1] = time[0], time[0]
            elif order in ['end']:
                event_time[2], event_time[3] = time[1], time[1]
            return event_time
        else:
            return event_time

    @staticmethod
    def inferTimeByOper(event_time, time, oper):

        if not time:
            return event_time

        if len(time) == 2:
            time = [time[0], time[1], None, None]
        oper_new = [oper[0:4], oper[4:8], oper[8:12], oper[12:16]]
        for index_t, oper_t in enumerate(oper_new):
            for index_e, oper_e in enumerate(list(oper_t)):
                if int(oper_e):
                    event_time[index_e] = time[index_t]
        return event_time



    @staticmethod
    def inferTimeOfEvent(tlink_list, oper=False, verbose=0):
        event_time = [None, None, None, None]
        for tlink in tlink_list:
            if verbose:
                print(event_time, tlink.targ.value, tlink.targ.tanchor, tlink.rel)
            event_time = TimeInference.inferTimeByOrder(event_time, tlink.targ.tanchor, tlink.rel) if not oper else \
                         TimeInference.inferTimeByOper(event_time, tlink.targ.tanchor, tlink.rel)
        print()
        return event_time


def main():
    sent_win = 10
    oper = True
    timeml_dir = os.path.join(os.path.dirname(__file__), "data/Timebank")
    anchor_file = os.path.join(os.path.dirname(__file__), "data/event-times_normalized.tab")
    pkl_file = os.path.join(os.path.dirname(__file__), "data/unittest-%s-%s_w%i_%s.pkl" % (timeml_dir.split('/')[-1],
                                                                                           anchor_file.split('/')[-1],
                                                                                           sent_win,
                                                                                           'oper' if oper else 'order'))

    anchor_file2doc(timeml_dir, anchor_file, pkl_file, sent_win, oper=True)

    doc_dic = load_doc(pkl_file)

    single_correct, multiple_correct, single, multiple = 0, 0, 0, 0

    for doc_id, doc in doc_dic.items():
        if doc_id in TBD_TEST:
            for event_id, event in doc.events.items():
                # print(event_id, event.value)
                tlinks = doc.getTlinkListByMention(event_id)
                event_time = TimeInference.inferTimeOfEvent(tlinks, oper=True, verbose=1)
                # if event_time and event_time[2] is None and event_time[3] is None:
                #     event_time = [event_time[0], event_time[1]]
                if event.tanchor:
                    if event.tanchor[2] is None and event.tanchor[3] is None:
                        single += 1
                    elif len(event.tanchor) == 4:
                        multiple += 1
                    if event_time == list(event.tanchor) if event.tanchor else None:
                        if event.tanchor[2] is None and event.tanchor[3] is None:
                            single_correct += 1
                        else:
                            multiple_correct += 1
                if not (event_time == list(event.tanchor) if event.tanchor else None):
                    print("event time: pred %s, gold %s %s" % (str(event_time), event.value, (event_time == list(event.tanchor) if event.tanchor else None)))
                    print()
    print("single num %i, multiple num %i" % (single, multiple))
    print("single accuracy %.3f, multiple accurracy %.3f" % (single_correct/single, multiple_correct/multiple))

if __name__ == '__main__':
    main()
