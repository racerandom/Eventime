# from TempObject import Mention,Event,Timex
# from TempNormalization import *
# import TempUtils
#
# from nltk import sent_tokenize, word_tokenize


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



