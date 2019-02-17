# coding=utf-8

from stanfordcorenlp import StanfordCoreNLP
from nltk.parse.dependencygraph import DependencyGraph, malt_demo
import networkx as nx
import json
import warnings

warnings.simplefilter("ignore", ResourceWarning)

def reformSDPforMention(mention_tok_ids, sdp, direct='mention2root'):
    if direct == 'mention2root':
        return mention_tok_ids + [tok for tok in sdp if tok not in mention_tok_ids]
    elif direct == 'root2mention':
        return [tok for tok in sdp if tok not in mention_tok_ids] + mention_tok_ids
    else:
        raise Exception("[Error] Unknown direct for reform SDP for a Mention!!!")


def nxGraphWroot(dep_graph):
    """Convert the data in a ``nodelist`` into a networkx labeled directed graph.
        Include the ROOT node
    """
    import networkx

    nx_nodelist = list(range(0, len(dep_graph.nodes))) ##
    nx_edgelist = [
        (n, dep_graph._hd(n), dep_graph._rel(n))
        for n in nx_nodelist
    ]
    dep_graph.nx_labels = {}
    for n in nx_nodelist:
        dep_graph.nx_labels[n] = dep_graph.nodes[n]['word']

    g = networkx.MultiDiGraph()
    g.add_nodes_from(nx_nodelist)
    g.add_edges_from(nx_edgelist)

    return g


class TempSyntax():

    def __init__(self):
        self.nlp_server = StanfordCoreNLP('http://localhost', port=9000)

    def get_dep_graph(self,
                      sent,
                      dep_ver='SD'  ## 'SD': stanford dependency, 'UD': universal dependency
                      ):
        props={'annotators': 'tokenize, ssplit, pos, depparse',
               'pipelineLanguage': 'en',
               'outputFormat': 'json',
               'depparse.model': "edu/stanford/nlp/models/parser/nndep/english_%s.gz" % dep_ver}
        dep_parse = self.nlp_server.annotate(sent, properties=props)
        dep_json = json.loads(dep_parse)
        dep_info_sent0 = dep_json['sentences'][0]['basicDependencies']
        tok_info_sent0 = dep_json['sentences'][0]['tokens']
        sorted_deps = sorted(dep_info_sent0, key=lambda x: x['dependent'])
        conll_str = ''.join(["%s\t%s\t%i\t%s\n" % (dep['dependentGloss'],
                                                   tok_info_sent0[dep['dependent'] - 1]['pos'],
                                                   dep['governor'],
                                                   dep['dep']) for dep in sorted_deps])

        return DependencyGraph(conll_str)

    def get_sdp(self,
                dep_graph,
                sour, ## sour token conll_id
                targ): ## targ token conll_id
        sd_dep_nx = nxGraphWroot(dep_graph).to_undirected()
        return nx.shortest_path(sd_dep_nx, source=sour, target=targ)

    def get_token(self, text):
        """
        :param text: a list of unprocessed sentence
        :param nlp_server: an initialized stanford corenlp server
        :return: a list of sentences information (split sentences, tokenized words, Part-of-Speech of tokens)
        """
        props = {'annotators': 'tokenize', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
        text_parse = self.nlp_server.annotate(text, properties=props)
        text_json = json.loads(text_parse)['tokens']
        return [ tok['originalText'] for tok in text_json]

    def get_sent(self, text):
        """
        :param text: a list of unprocessed sentence
        :param nlp_server: an initialized stanford corenlp server
        :return: a list of sentences information (split sentences, tokenized words, Part-of-Speech of tokens)
        """
        props = {'annotators': 'tokenize, ssplit', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
        text_parse = self.nlp_server.annotate(text, properties=props)
        text_json = json.loads(text_parse)['sentences']
        return [ ' '.join([ tok['originalText'] for tok in sent['tokens']]) for sent in text_json]

    def close(self):
        self.nlp_server.close()

    @staticmethod
    def get_deps_along_sdp(sdp, dep_graph):
        deps = []
        directs = []
        for i in range(len(sdp) - 1):
            deps.append(TempSyntax.get_dep_by_nodes(dep_graph, sdp[i], sdp[i + 1]))
        deps.append("%s-%s" % (dep_graph.nodes[sdp[-1]]['rel'], 'head'))
        return deps


    @staticmethod
    def get_dep_by_nodes(dep_graph, sour_id, targ_id):
        sour = dep_graph.nodes[sour_id]
        if sour['head'] == targ_id:
            return "%s-%s" % (sour['rel'], 'head')
        else:
            for dep, childs in sour['deps'].items():
                if targ_id in childs:
                    return "%s-%s" % (dep, 'modifier')
            raise Exception('[ERROR] There is no dep relation between %i and %i' % (sour_id, targ_id))

import unittest

class TestTempSyntax(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestTempSyntax, self).__init__(*args, **kwargs)
        self.TempSyntaxer = TempSyntax()
        # self.sent_case = "The panel also will look at the exodus of about 2 million Rwanda Hutus to neighboring countries where they lived in U.N.-run refugee camps for 2-1/2 years ."
        self.sent_case = "Steve Jobs was the co-founder and CEO of Apple and formerly Pixar ."

    def test_get_token(self):
        tokens = self.TempSyntaxer.get_token(self.sent_case)
        print(tokens)

    def test_get_sent(self):
        sents = self.TempSyntaxer.get_sent(self.sent_case)
        print(sents)

    def test_draw_dep_tree(self):
        sd_dep_graph = self.TempSyntaxer.get_dep_graph(self.sent_case, dep_ver='SD')
        sd_dep_graph.tree().draw()

    def test_get_dep_graph(self):
        sd_dep_graph = self.TempSyntaxer.get_dep_graph(self.sent_case, dep_ver='SD')
        print(sd_dep_graph)

    def test_get_dep_conll(self):
        sd_dep_graph = self.TempSyntaxer.get_dep_graph(self.sent_case, dep_ver='SD')
        print(sd_dep_graph.to_conll(4))

    def test_get_sdp(self):
        sd_dep_graph = self.TempSyntaxer.get_dep_graph(self.sent_case, dep_ver='SD')
        sd_sdp = self.TempSyntaxer.get_sdp(sd_dep_graph, 16, 0)
        print(sd_sdp)

    def test_get_deps_along_sdp(self):
        sd_dep_graph = self.TempSyntaxer.get_dep_graph(self.sent_case, dep_ver='SD')
        sd_sdp = self.TempSyntaxer.get_sdp(sd_dep_graph, 16, 25)
        print(sd_sdp)
        print(self.TempSyntaxer.get_deps_along_sdp(sd_sdp, sd_dep_graph))

    def test_get_dep_nx(self):
        sd_dep_graph = self.TempSyntaxer.get_dep_graph(self.sent_case, dep_ver='SD')
        sd_sdp = self.TempSyntaxer.get_sdp(sd_dep_graph, 0, 16)

        print(["%s %s %s" % (sd_dep_graph.nodes[node_index]['word'],
                             sd_dep_graph.nodes[node_index]['tag'],
                             sd_dep_graph.nodes[node_index]['rel']) for node_index in sd_sdp])

        for node_index in sd_sdp:
            node = sd_dep_graph.nodes[node_index]
            for rel, children in node['deps'].items():
                if rel in ['aux', 'case', 'mark', 'advmod']:
                    for child_id in children:
                        print(sd_dep_graph.nodes[child_id]['word'])
            print(node['word'])


if __name__ == '__main__':

    TempSyntaxer = TempSyntax()
    sd_dep_graph = TempSyntaxer.get_dep_graph(
        "Robots in popular culture are there to remind us of the awesomeness of unbound human agency.", dep_ver='UD')
    sd_dep_graph.tree().draw()

    # unittest.main()

# dep_networkx = nlp_server.get_dep_networkx(text, dep_ver='SD')
# print(nx.shortest_path(dep_networkx, source='chased5', target='2018-12'))


# from nltk.parse.corenlp import CoreNLPDependencyParser
#
# dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
# parse, = dep_parser.raw_parse(sentence)
# print(type(parse))
# print(parse.to_conll(4))