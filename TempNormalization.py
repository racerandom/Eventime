import unittest
import re
import warnings

warnings.simplefilter("ignore", ResourceWarning)

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def dayLengthOfMention(tanchor):
    if not tanchor:
        return 'Unknown'
    elif len(tanchor) == 2:
        return 'SingleDay'
    elif len(tanchor) == 4:
        return 'MultiDay'
    else:
        return 'Unknown'


def last_day_of_month(date):
    if date.month == 12:
        return date.replace(day=31)
    return date.replace(month=date.month+1, day=1) + timedelta(days=-1)


def regular_season(value):
    year = value.split('-')[0]
    season = value.split('-')[1]
    if season in ['SP']:
        begin = datetime.strptime("%s-03-21" % year, '%Y-%m-%d')
    elif season in ['SU']:
        begin = datetime.strptime("%s-06-21" % year, '%Y-%m-%d')
    elif season in ['F', 'FA', 'AU']:
        begin = datetime.strptime("%s-09-21" % year, '%Y-%m-%d')
    elif season in ['W', 'WI']:
        begin = datetime.strptime("%s-12-21" % year, '%Y-%m-%d')
    end = begin + relativedelta(months=3, days=-1)
    return begin, begin, end, end


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
    end = begin + relativedelta(months=3, days=-1)
    return begin, begin, end, end


def quarter_delta(date, num):
    currQuarter = int((date.month - 1) / 3 + 1)
    # print(currQuarter, 3 * currQuarter - 2, 3 * currQuarter + 1, 1)
    dtFirstDay = datetime(date.year, 3 * currQuarter - 2, 1)
    dtLastDay = datetime(date.year, 3 * currQuarter, 1) + relativedelta(months=1, days=-1)
    if num >= 0:
        begin = dtFirstDay + relativedelta(months=(1 if num > 1 else 0) * 3)
        end = dtLastDay + relativedelta(months=(num + 1) * 3)
    else:
        begin = dtFirstDay + relativedelta(months=num * 3)
        end = dtLastDay + relativedelta(months=-3)
    return begin, begin, end, end


def normalize_season_day(day_value):
    year = day_value.split('-')[0]
    season = day_value.split('-')[1]
    day_offset = day_value.split('-')[2]
    if season in ['SP']:
        norm_begin = datetime.strptime("%s-03-21" % year, '%Y-%m-%d')
    elif season in ['SU']:
        norm_begin = datetime.strptime("%s-06-21" % year, '%Y-%m-%d')
    elif season in ['F', 'FA', 'AU']:
        norm_begin = datetime.strptime("%s-09-21" % year, '%Y-%m-%d')
    elif season in ['W', 'WI']:
        norm_begin = datetime.strptime("%s-12-21" % year, '%Y-%m-%d')
    else:
        raise Exception('[ERROR] Fail to recognize season string for normalization!!!', season)

    if day_offset == 'FIRST':
        return norm_begin
    elif day_offset == 'SECOND':
        return norm_begin + relativedelta(days=1)
    elif day_offset == 'LAST':
        return norm_begin + relativedelta(months=3, days=-1)
    else:
        raise Exception('[ERROR] Fail to recognize day offset for normalization!!!', day_offset)


def normalize_quarter_day(day_value):
    year = day_value.split('-')[0]
    quarter = day_value.split('-')[1]
    day_offset = day_value.split('-')[2]
    if quarter in ['Q1']:
        norm_begin = datetime.strptime("%s-01-01" % year, '%Y-%m-%d')
    elif quarter in ['Q2']:
        norm_begin = datetime.strptime("%s-04-01" % year, '%Y-%m-%d')
    elif quarter in ['Q3']:
        norm_begin = datetime.strptime("%s-07-01" % year, '%Y-%m-%d')
    elif quarter in ['Q4']:
        norm_begin = datetime.strptime("%s-10-01" % year, '%Y-%m-%d')
    else:
        raise Exception('[ERROR] Fail to recognize quarter string for normalization!!!', quarter)

    if day_offset == 'FIRST':
        return norm_begin
    elif day_offset == 'SECOND':
        return norm_begin + relativedelta(days=1)
    elif day_offset == 'LAST':
        return norm_begin + relativedelta(months=3, days=-1)
    else:
        raise Exception('[ERROR] Fail to recognize day flag for normalization!!!', day_offset)


def normalize_month_day(day_value):
    year = day_value.split('-')[0]
    month = day_value.split('-')[1]
    day_offset = day_value.split('-')[2]
    norm_begin = datetime.strptime("%s-%s" % (year, month), '%Y-%m')

    if day_offset in ['FIRST']:
        return norm_begin
    elif day_offset == 'SECOND':
        return norm_begin + relativedelta(days=1)
    elif day_offset == 'LAST':
        return norm_begin + relativedelta(months=1, days=-1)
    else:
        raise Exception('[ERROR] Fail to recognize day_offset for normalization!!!', day_offset)


def normalize_week_day(day_value):
    year = day_value.split('-')[0]
    week = int(day_value.split('-')[1][1:]) - 1
    day_offset = day_value.split('-')[2]
    norm_begin = datetime.strptime("%s-W%i-0" % (year, week), "%Y-W%W-%w")

    if day_offset in ['FIRST']:
        return norm_begin
    elif day_offset == 'SECOND':
        return norm_begin + relativedelta(days=1)
    elif day_offset == 'LAST':
        return norm_begin + relativedelta(weeks=1, days=-1)
    else:
        raise Exception('[ERROR] Fail to recognize day_offset for normalization!!!', day_offset)


def normalize_year_day(day_value, delta=1):
    year = day_value.split('-')[0]
    day_offset = day_value.split('-')[1]
    norm_begin = datetime.strptime("%s" % (year), "%Y")
    if day_offset in ['FIRST']:
        return norm_begin
    elif day_offset == 'SECOND':
        return norm_begin + relativedelta(days=1)
    elif day_offset == 'LAST':
        return norm_begin + relativedelta(years=delta, days=-1)
    else:
        raise Exception('[ERROR] Fail to recognize day_offset for normalization!!!', day_offset)


def normalize_event_day(day_value):

    day_value = day_value.replace('-EAR', '').replace('-LAT', '').replace('-MID', '')

    if re.match(r'\d{4}-\d{1,2}-\d{1,2}$', day_value):
        norm_day = datetime.strptime(day_value, '%Y-%m-%d')
    elif re.match(r'\d{4}-\d{1,2}-(FIRST|LAST)', day_value):
        norm_day = normalize_month_day(day_value)
    elif re.match(r'\d{4}-(FIRST|LAST)', day_value):
        norm_day = normalize_year_day(day_value)
    elif re.match(r'\d{3}X-(FIRST|LAST)', day_value):
        norm_day = normalize_year_day(day_value.replace('X', '0'), delta=10)
    elif day_value.split('-')[1] in ['SP', 'SU', 'F', 'FA', 'AU', 'W', 'WI']:
        norm_day = normalize_season_day(day_value)
    elif re.match(r'\d{4}-[Ww]\d{1,2}-', day_value):
        norm_day = normalize_week_day(day_value)
    elif re.match(r'\d{4}-[Qq]\d{1}-', day_value):
        norm_day = normalize_quarter_day(day_value)
    else:
        raise Exception('[ERROR] Fail to recognize a event day for normalization!!!', day_value)
    return norm_day


# normalize tanchor value of EVENT
def normalize_tanchor(value):

    def normalize_single_tanchor(value):
        value = value.strip()
        singlematch = re.compile("\(after ([^']+), before ([^']+)\)")
        if re.match(singlematch, value):
            singleout = singlematch.findall(value)
            # if re.match(r'\d{4}-\d{1,2}-\d{1,2}$', singleout[0][0]):
            #     after = datetime.strptime(singleout[0][0], '%Y-%m-%d')
            # else:
            #     after, after = normalize_time(singleout[0][0])
            after = normalize_event_day(singleout[0][0])

            # if re.match(r'\d{4}-\d{1,2}-\d{1,2}$', singleout[0][1]):
            #     before = datetime.strptime(singleout[0][1], '%Y-%m-%d')
            # else:
            #     before, before = normalize_time(singleout[0][1])
            before = normalize_event_day(singleout[0][1])

            return after, before
        elif 'after' in value:
            value = value.strip('after ')
            # if re.match(r'\d{4}-\d{1,2}-\d{1,2}$', value):
            #     return datetime.strptime(value, '%Y-%m-%d'), None
            # else:
            #     after, after = normalize_time(value)
            after = normalize_event_day(value)

            return after, None
        elif 'before' in value:
            value = value.strip('before ')
            # if re.match(r'\d{4}-\d{1,2}-\d{1,2}$', value):
            #     return None, datetime.strptime(value, '%Y-%m-%d')
            # else:
            #     before, before = normalize_time(value)
            #     return None, before
            before = normalize_event_day(value)
            return None, before

        else:
            norm_day = normalize_event_day(value)
            return norm_day, norm_day
        # else:
        #     ## temporal code
        #     # ba, bb, ea, eb = normalize_time(value)
        #     # if point == 'begin':
        #     #     return ba, ba
        #     # elif point == 'end':
        #     #     return eb, eb
        #     # else:
        #     #     return ba, bb, ea, eb
        #     # return normalize_time(value)
        #     raise Exception('[ERROR] event time normalization!!!', value)

    def normalize_multi_tanchor(value):
        if 'freq' in value:
            multimatch = re.compile(r"\(begin:(.+), end:(.+), freq:(.+)\)")
        elif 'dur' in value:
            multimatch = re.compile(r"\(begin:(.+), end:(.+), dur:(.+)\)")
        elif 'dis' in value:
            multimatch = re.compile(r"\(begin:(.+), end:(.+), dis=(.+)\)")
        else:
            multimatch = re.compile(r"\(begin:(.+), end:(.+)\)")

        if re.match(multimatch, value):
            mout = multimatch.search(value)
            ba, bb = normalize_single_tanchor(mout.group(1))
            ea, eb = normalize_single_tanchor(mout.group(2))
            if not bb:
                bb = eb
            if not ea:
                ea = ba
            return ba, bb, ea, eb
        else:
            return None

    if 'AND' in value or 'OR' in value:
        return None

    # print(value)

    if re.match(r"\(begin:(.+), end:(.+)\)", value):
        norm_tanchor = normalize_multi_tanchor(value)
    else:
        norm_ba, norm_bb = normalize_single_tanchor(value)
        norm_tanchor = (norm_ba, norm_bb, norm_ba, norm_bb)
    return norm_tanchor


# normalize anchor of Event (Reimer 2016 data)
def normalize_anchor(anchor):

    def normalize_single_anchor(anchor):
        if re.match(r"\d{4}-\d{1,2}-\d{1,2}", anchor):
            match_out = re.compile(r"\d{4}-\d{1,2}-\d{1,2}").findall(anchor)
            after = datetime.strptime(match_out[0], '%Y-%m-%d')
            before = after
        elif re.match(r"after(\d{4}-\d{1,2}-\d{1,2})before(\d{4}-\d{1,2}-\d{1,2})", anchor):
            match_out = re.compile(r"after(\d{4}-\d{1,2}-\d{1,2})before(\d{4}-\d{1,2}-\d{1,2})").findall(anchor)
            after = datetime.strptime(match_out[0][0], '%Y-%m-%d')
            before = datetime.strptime(match_out[0][1], '%Y-%m-%d')
        elif re.match(r"after(\d{4}-\d{1,2}-\d{1,2})", anchor):
            match_out = re.compile(r"after(\d{4}-\d{1,2}-\d{1,2})").findall(anchor)
            after = datetime.strptime(match_out[0], '%Y-%m-%d')
            before = None
        elif re.match(r"before(\d{4}-\d{1,2}-\d{1,2})", anchor):
            match_out = re.compile(r"before(\d{4}-\d{1,2}-\d{1,2})").findall(anchor)
            after = None
            before = datetime.strptime(match_out[0], '%Y-%m-%d')
        else:
            print("Cannot normalize single-day anchor:", anchor)
            return None

        return after, before

    if not re.match(r"beginPoint=(.+)endPoint=(.+)", anchor):
        anchor = normalize_single_anchor(anchor)
    else:
        match_out = re.compile(r"beginPoint=(.+)endPoint=(.+)").findall(anchor)
        ba, bb = normalize_single_anchor(match_out[0][0])
        ea, eb = normalize_single_anchor(match_out[0][1])
        anchor = ba, bb, ea, eb
    return anchor


# return 4-value tuple by normalizing timex value input
def normalize_time(value):
    value = value.strip().split('T')[0]
    if value[0].isdigit():
        if re.match(r'\d{4}-\d{1,2}-\d{1,2}$', value): # YYYY-MM-DD
            return datetime.strptime(value, '%Y-%m-%d'), datetime.strptime(value, '%Y-%m-%d')
        # elif value.count('-') == 2 and value.split('-')[1][0] == 'W' and value.split('-')[2] == "WE":
        elif re.match(r'\d{4}-[Hh]\d{1}$', value): # YYYY-HN
            if value[-1] == '1':
                begin = datetime.strptime(value.split('-')[0], '%Y')
                end = begin + relativedelta(months=6, days=-1)
            elif value[-1] == '2':
                begin = datetime.strptime(value.split('-')[0], '%Y') + relativedelta(months=6)
                end = begin + relativedelta(months=6, days=-1)
            return begin, begin, end, end
        elif re.match(r'\d{4}-\d{1,2}$', value): # YYYY-MM
            begin = datetime.strptime(value, '%Y-%m')
            end = last_day_of_month(begin)
            return begin, begin, end, end
        elif re.match(r'\d{4}-[Ww]\d{1,2}$', value): # YYYY-WN
            year = value.split('-')[0]
            week = int(value.split('-')[1][1:]) - 1
            begin = datetime.strptime("%s-W%i-0" % (year, week), "%Y-W%W-%w")
            end = begin + timedelta(days=6)
            return begin, begin, end, end
        elif re.match(r'\d{4}-[Ww]\d{1,2}-WE$', value): # YYYY-WN
            year = value.split('-')[0]
            week = int(value.split('-')[1][1:]) - 1
            begin = datetime.strptime("%s-W%i-0" % (year, week), "%Y-W%W-%w") + timedelta(days=6)
            end = begin + timedelta(days=1)
            return begin, begin, end, end
        elif re.match(r'\d{4}-[Qq]\d{1}$', value): # YYYY-QN
            return regular_quarter(value)
        elif value.split('-')[-1] in ['SP', 'SU', 'F', 'FA', 'AU', 'W', 'WI']:
        # elif re.match(r'\d{4}-\bSP\b|\bSU\b|\bF\b|\bFA\b|\bW\b|\bWI\b$', value): # YYYY-SU
            return regular_season(value)
        elif re.match(r'\d{4}$', value):
            begin = datetime.strptime(value, '%Y')
            end = begin + relativedelta(months=12, days=-1)
            return begin, begin, end, end
        elif re.match(r'\d{3}$', value):
            begin = datetime.strptime("%s0" % value, '%Y')
            end = datetime.strptime("%s9" % value, '%Y') + relativedelta(months=12, days=-1)
            return begin, begin, end, end
        elif re.match(r'\d{4}-\d{1,2}-(EAR|MID|LAT)-(FIRS|LAS)', value):  # YYYY-MM
            y, m, p, d = value.split('-')
            if p == 'EAR':
                if d == 'FIRS':
                    begin = datetime.strptime('%s-%s' % (y, m), '%Y-%m')
                elif d == 'LAS':
                    begin = datetime.strptime('%s-%s' % (y, m), '%Y-%m') + relativedelta(days=+7)
            return begin, begin
        elif re.match(r'\d{4}-(EAR|MID|LAT)-(FIRS|LAS)', value):  # YYYY-MM-EAR|MID|LAT-FIRST|LAST
            y, p, d = value.split('-')
            if p == 'EAR':
                if d == 'FIRST':
                    begin = datetime.strptime(y, '%Y')
                elif d == 'LAS':
                    begin = datetime.strptime(y, '%Y') + relativedelta(months=3, days=-1)
            return begin, begin
        elif re.match(r'\d{4}-(SP|SU|AU|WI)-(FIRS|LAS)', value):  # YYYY-SP|SU|AU|WI-FIRST|LAST
            y, p, d = value.split('-')
            if p in ['FA', 'AU']:
                if d == 'FIRST':
                    begin = datetime.strptime("%s-09-21" % y, '%Y')
                elif d == 'LAS':
                    begin = datetime.strptime("%s-09-21" % y, '%Y') + relativedelta(months=3, days=-1)
            return begin, begin
        elif re.match(r'\d{4}-\d{1,2}-XX', value):
            y, m, d = value.split('-')
            begin = datetime.strptime('%s-%s' % (y, m), '%Y-%m')
            end = begin + relativedelta(months=1, days=-1)
            return begin, begin, end, end
        elif re.match(r'\d{4}-XX', value):
            y, m = value.split('-')[0], value.split('-')[1]
            begin = datetime.strptime('%s' % (y), '%Y')
            end = begin + relativedelta(years=1, days=-1)
            return begin, begin, end, end
        elif re.match(r'\d{3}X', value):
            y = value.split('-')[0]
            begin = datetime.strptime('%s' % (y.replace('X', '0')), '%Y')
            end = begin + relativedelta(years=10, days=-1)
            return begin, begin, end, end
        elif re.match(r'\d{2}XX', value):
            y = value.split('-')[0]
            begin = datetime.strptime('%s' % (y.replace('X', '0')), '%Y')
            end = begin + relativedelta(years=100, days=-1)
            return begin, begin, end, end

        print("[cannot normalize time] time value:", value)
    return None


def normalize_relative(timex, relative_timex):
    if not relative_timex.tanchor:
        return None
    else:
        if timex.value[0] == 'P' and timex.value[1:-1].isdigit() and timex.value[-1] in ['Y', 'M', 'W', 'D']:
            if timex.endPoint:
                end = relative_timex.tanchor[-1]
                if timex.value[-1] == 'Y':
                    begin = end + relativedelta(years=-int(timex.value[1:-1]), days=1)
                elif timex.value[-1] == 'M':
                    begin = end + relativedelta(months=-int(timex.value[1:-1]), days=1)
                elif timex.value[-1] == 'W':
                    begin = end + relativedelta(weeks=-int(timex.value[1:-1]), days=1)
                elif timex.value[-1] == 'D':
                    begin = end + relativedelta(days=-int(timex.value[1:-1]) + 1)
                return begin, begin, end, end
            elif timex.beginPoint:
                begin = relative_timex.tanchor[0]
                if timex.value[-1] == 'Y':
                    end = begin + relativedelta(years=int(timex.value[1:-1]), days=-1)
                elif timex.value[-1] == 'M':
                    end = begin + relativedelta(months=int(timex.value[1:-1]), days=-1)
                elif timex.value[-1] == 'W':
                    end = begin + relativedelta(weeks=int(timex.value[1:-1]), days=-1)
                elif timex.value[-1] == 'D':
                    end = begin + relativedelta(days=int(timex.value[1:-1]) - 1)
                return begin, begin, end, end
        elif re.match(r'\d{4}-Q[A-Z]$', timex.value):
            # print(relative_timex.value)
            date = datetime.strptime(relative_timex.value, '%Y-%m-%d')
            # print(date)
            # print(quarter_delta(date, -1))
            return quarter_delta(date, -1)
        elif timex.value == "PAST_REF":
            return None, relative_timex.tanchor[-1]
        elif timex.value == "FUTURE_REF":
            return relative_timex.tanchor[0], None
        elif timex.value == "PRESENT_REF":
            return relative_timex.tanchor[0], relative_timex.tanchor[1]
        else:
            return None


class TestTempNormalization(unittest.TestCase):

    def test_normalize_single_anchor(self):
        anchor0 = "1990-09-01"
        anchor1 = "after1990-09-01"
        anchor2 = "before1990-09-28"
        anchor3 = "after1990-09-01before1990-09-28"
        print(anchor0, normalize_anchor(anchor0))
        print(anchor1, normalize_anchor(anchor1))
        print(anchor2, normalize_anchor(anchor2))
        print(anchor3, normalize_anchor(anchor3))

    def test_normalize_multi_anchor(self):
        anchor1 = "beginPoint=after1998-02-01before1998-02-04endPoint=after1998-02-05"
        anchor2 = "beginPoint=1998-02-05endPoint=1998-02-06"
        anchor3 = "beginPoint=after1998-02-01before1998-02-04endPoint=after1998-02-05before1998-03-06"
        anchor4 = "beginPoint=1994-04-06endPoint=before1994-12-31"
        print(anchor1, normalize_anchor(anchor1))
        print(anchor2, normalize_anchor(anchor2))
        print(anchor3, normalize_anchor(anchor3))
        print(anchor4, normalize_anchor(anchor4))

    def test_normalize_tanchor(self):
        tanchor1 = "(after 1999-11-LAT-FIRST, after 2000-04-03)"
        tanchor1 = "(begin:(after 1953-01-01, before 1953-12-31), end:(after 1953-01-01, before 1953-12-31))"
        print(tanchor1)
        print(normalize_tanchor(tanchor1))
