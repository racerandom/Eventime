import unittest
import re


from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta


def last_day_of_month(date):
    if date.month == 12:
        return date.replace(day=31)
    return date.replace(month=date.month+1, day=1) + timedelta(days=-1)


# return 4-tuple out from time value input
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


# return 4-tuple out from time value input
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


# normalize tanchor value in EVENT
def normalize_tanchor(value):

    def normalize_single_tanchor(value, point='certain'):
        singlematch = re.compile("\(after ([^']+), before ([^']+)\)")
        if re.match(singlematch, value):
            singleout = singlematch.findall(value)
            if re.match(r'\d{4}-\d{1,2}-\d{1,2}$', singleout[0][0]):
                after = datetime.strptime(singleout[0][0], '%Y-%m-%d')
            else:
                ba, bb, ea, after = normalize_time(singleout[0][0])
            if re.match(r'\d{4}-\d{1,2}-\d{1,2}$', singleout[0][1]):
                before = datetime.strptime(singleout[0][1], '%Y-%m-%d')
            else:
                before, bb, ea, eb = normalize_time(singleout[0][1])
            return after, before
        elif 'after' in value:
            value = value.strip('after ')
            if re.match(r'\d{4}-\d{1,2}-\d{1,2}$', value):
                return datetime.strptime(value, '%Y-%m-%d'), None
            else:
                ba, bb, ea, eb = normalize_time(value)
                return eb, None
        elif 'before' in value:
            value = value.strip('before ')
            if re.match(r'\d{4}-\d{1,2}-\d{1,2}$', value):
                return None, datetime.strptime(value, '%Y-%m-%d')
            else:
                ba, bb, ea, eb = normalize_time(value)
                return None, ba

        elif re.match(r'\d{4}-\d{1,2}-\d{1,2}$', value):
            after = datetime.strptime(value, '%Y-%m-%d')
            return after, after
        else:
            ## temporal code
            ba, bb, ea, eb = normalize_time(value)
            if point == 'begin':
                return ba, ba
            elif point == 'end':
                return eb, eb
            else:
                return ba, bb, ea, eb

    def normalize_multi_tanchor(value):
        # print(value)
        if 'freq' in value:
            multimatch = re.compile("\(begin:(.+), end:(.+), freq:(.+)\)")
        elif 'dur' in value:
            multimatch = re.compile("\(begin:(.+), end:(.+), dur:(.+)=\)")
        else:
            multimatch = re.compile("\(begin:(.+), end:(.+)\)")
        if re.match(multimatch, value):
            mout = multimatch.search(value)
            ba, bb = normalize_single_tanchor(mout.group(1), 'begin')
            ea, eb = normalize_single_tanchor(mout.group(2), 'end')
            return ba, bb, ea, eb


    if 'AND' in value or 'OR' in value:
        return None
    else:
        if 'dis=' in value:
            return None
        if re.match(r"\(begin:(.+), end:(.+)\)", value):
            return normalize_multi_tanchor(value)
        else:
            return normalize_single_tanchor(value)



# normalize values in TIMEXs to tanchor tuples
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