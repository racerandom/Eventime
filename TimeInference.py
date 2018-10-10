# coding=utf-8

import os
from TimeMLReader import *
from TempData import *


def day_length(time):
    if not time:
        return None
    elif time[0] == time[1]:
        return 'single'
    else:
        return 'multiple'

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
                    event_time[0], event_time[1], event_time[3], event_time[4] = time[0], time[0], time[2], time[2]
            elif order in ['after_single']:
                updateBeginAfter(event_time, time[1])
            elif order in ['after_multiple']:
                updateBeginAfter(event_time, time[1])
                updateEndAfter(event_time, time[1])
            elif order in ['before_single']:
                updateBeginBefore(event_time, time[0])
            elif order in ['before_multiple']:
                updateBeginBefore(event_time, time[0])
                updateEndBefore(event_time, time[0])
            elif order in ['include']:
                updateBeginBefore(event_time, time[0])
                updateEndAfter(event_time, time[1])
            elif order in ['is_included_single']:
                updateBeginAfter(event_time, time[0])
                updateBeginBefore(event_time, time[2])
            elif order in ['is_included_multiple']:
                updateBeginAfter(event_time, time[0])
                updateEndBefore(event_time, time[1])
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
    def inferTimeOfEvent(tlink_list, verbose=0):
        event_time = [None, None, None, None]
        for tlink in tlink_list:
            if verbose:
                print(event_time, tlink.targ.value, tlink.targ.tanchor, tlink.rel)
            event_time = TimeInference.inferTimeByOrder(event_time, tlink.targ.tanchor, tlink.rel)

        return event_time


def main():
    sent_win = 4
    
    timeml_dir = os.path.join(os.path.dirname(__file__), "data/Timebank")
    anchor_file = os.path.join(os.path.dirname(__file__), "data/event-times_normalized.tab")
    pkl_file = os.path.join(os.path.dirname(__file__), "data/unittest-%s-%s_w%i.pkl" % (timeml_dir.split('/')[-1], anchor_file.split('/')[-1], sent_win))

    # anchor_file2doc(timeml_dir, anchor_file, pkl_file, sent_win, oper=False)

    doc_dic = load_doc(pkl_file)

    single_correct, multiple_correct, single, multiple = 0, 0, 0, 0

    for doc_id, doc in doc_dic.items():
        if doc_id in TBD_TEST:
            for event_id, event in doc.events.items():
                # print(event_id, event.value)
                tlinks = doc.getTlinkListByMention(event_id)
                event_time = TimeInference.inferTimeOfEvent(tlinks, verbose=1)
                if event_time and not event_time[2] and not event_time[3]:
                    event_time = [event_time[0], event_time[1]]
                if event.tanchor:
                    if len(event.tanchor) == 2:
                        single += 1
                    elif len(event.tanchor) == 4:
                        multiple += 1
                    if event_time == list(event.tanchor) if event.tanchor else None:
                        if len(event.tanchor) == 2:
                            single_correct += 1
                        elif len(event.tanchor) == 4:
                            multiple_correct += 1
                if not (event_time == list(event.tanchor) if event.tanchor else None):
                    print("event time: pred %s, gold %s %s" % (str(event_time), event.value, (event_time == list(event.tanchor) if event.tanchor else None)))
                    print()
    print("single num %i, multiple num %i, single accuracy %.3f, multiple accurracy %.3f" % (single, multiple, single_correct/single, multiple_correct/multiple))

if __name__ == '__main__':
    main()
