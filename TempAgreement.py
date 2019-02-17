# coding=utf-8
import os
import xml.etree.ElementTree as ET
from TempNormalization import *
from krippendorff_alpha import *

import unittest


def readEventFromTimeml(timeml_file):

    event_dic = {}

    root = ET.parse(timeml_file).getroot()
    for elem in root.iter():
        if elem.tag == 'TEXT':
            for mention_elem in elem:
                if mention_elem.tag == "EVENT":
                    event_key = "%s-%s" % (mention_elem.attrib['eid'], mention_elem.text)
                    if 'tanchor' in mention_elem.attrib:
                        event_dic[event_key] = mention_elem.attrib['tanchor']
                    else:
                        event_dic[event_key] = None
    return event_dic


# verbose: print disagreed anchors or not
def evalAgreementFromDir(annotator1_dir, annotator2_dir, norm=True, verbose=0):

    annotator1_dir = os.path.join(os.path.dirname(__file__), annotator1_dir)
    annotator2_dir = os.path.join(os.path.dirname(__file__), annotator2_dir)

    common_files = list(set(os.listdir(annotator1_dir)) & set(os.listdir(annotator2_dir)))

    print(len(os.listdir(annotator1_dir)), len(os.listdir(annotator2_dir)), len(common_files))

    agree_num, agree_single, single_num, multi_num, begin_agree_num, end_agree_num, total_num = 0, 0, 0, 0, 0, 0, 0
    multi_single = 0

    annotator1_out, annotator2_out = [], []

    for filename in common_files:

        an, tn = 0, 0

        annotator1_file = os.path.join(annotator1_dir, filename)
        annotator2_file = os.path.join(annotator2_dir, filename)

        print("Processing timeml file:", filename)

        annotator1_events = readEventFromTimeml(annotator1_file)
        annotator2_events = readEventFromTimeml(annotator2_file)

        common_event_ids = list(set(annotator1_events.keys()) & set(annotator2_events.keys()))

        total_num += len(common_event_ids)

        for event_id in sorted(common_event_ids, key=lambda x: int(x.split('-')[0][1:])):

            tn += 1

            if norm:
                if annotator1_events[event_id]:
                    try:
                        event1 = normalize_tanchor(annotator1_events[event_id])
                    except Exception as ex:
                        event1 = None
                else:
                    event1 = None
                if annotator2_events[event_id]:
                    try:
                        event2 = normalize_tanchor(annotator2_events[event_id])
                    except Exception as ex:
                        event2 = None
                else:
                    event2 = None
            else:
                event1 = annotator1_events[event_id].strip() if annotator1_events[event_id] else None
                event2 = annotator2_events[event_id].strip() if annotator2_events[event_id] else None

            if event1:
                annotator1_out.append(" ".join([ str(t) for t in event1]))
            else:
                annotator1_out.append(None)

            if event2:
                annotator2_out.append(" ".join([ str(t) for t in event2]))
            else:
                annotator2_out.append(None)

            if not event1 or not event2:
                continue
            elif event1 == event2:
                agree_num += 1
                an += 1
                if len(event1) == 2:
                    agree_single += 1
                # if verbose:
                #     print("[Agreed]", event_id, ":", annotator1_events[event_id], " ||| ", annotator2_events[event_id])
            else:
                if verbose:
                    print("[Disagreed]", event_id, ":", annotator1_events[event_id], " ||| ", annotator2_events[event_id])
                if len(event1) == 4 and len(event2) == 4:

                    multi_num += 1
                    if event1[0] == event2[0] and event1[1] == event2[1]:
                        begin_agree_num += 1
                    elif event1[2] == event2[2] and event1[3] == event2[3]:
                        end_agree_num += 1
                elif len(event1) == 2 and len(event2) == 2:
                    single_num += 1
                else:
                    if len(event1) == 2:
                        single = event1
                        multi = event2
                    else:
                        single = event2
                        multi = event1
                    if single[0] == multi[0] and single[-1] == multi[-1]:
                        multi_single += 1


        print('agreement per doc: %.4f' % (an / tn), '\n')

    print("krippendorff_alpha (nominal metric): %.3f" % krippendorff_alpha([annotator1_out, annotator2_out],
                                                                           nominal_metric,
                                                                           convert_items=str,
                                                                           missing_items=None))

    print("Agreed number: %i \n" % agree_num,
          "Agreed single: %i \n" % agree_single,
          "Agreed multi: %i \n" % (agree_num - agree_single),
          "Disagreed number: %i \n" % (total_num - agree_num),
          "-Both single (Disagreed): %i \n" % single_num,
          "-Both multi (Disagreed): %i \n" % multi_num,
          "--Agreed begin (Disagreed multi): %i \n" % begin_agree_num,
          "--Agreed end  (Disagreed multi): %i \n" % end_agree_num,
          "-Other (Disagreed): %i \n" % (total_num - agree_num - single_num - multi_num),
          "-Other (single/multi): %i \n" % multi_single,
          "Total: %i \n" % total_num,
          "Total disagreement: %.4f \n" % ( 1 - agree_num/ total_num))


def main():
    annotator1_dir = "/Users/fei-c/Resources/timex/anchor/final_1016/Agreement/Kd_Agreement/tml"
    annotator2_dir = "/Users/fei-c/Resources/timex/anchor/final_1016/Agreement/Td_Agreement/tml"
    evalAgreementFromDir(annotator1_dir, annotator2_dir, norm=True, verbose=True)


if __name__ == '__main__':
    main()


class TestTempAgreement(unittest.TestCase):

    def test_readEventFromTimeml(self):
        print(readEventFromTimeml("/Users/fei-c/Resources/timex/20180919/kd/AQA001_APW19980807.0261.tml"))