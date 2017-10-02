# python3
import pandas as pd
from pprint import pprint
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import smart_open, os, argparse

parser = argparse.ArgumentParser(description='Load a gensim model and query it.')
parser.add_argument('-i', '--input', metavar='INPUT', type=str, help='input file')
parser.add_argument('-o', '--out', metavar='OUT', type=str, help='outputfile name (example: "model.gensim")')
parser.add_argument('-t', '--type', metavar='TYPE', type=str, help='embedding type: "cbow" or "skipgram"')
args = parser.parse_args()

if not os.path.exists('./data_util/ready_data/'):
    os.makedirs('./data_util/ready_data/')

filenames = ['./data_util/ready_data/tweets_df_echo_ssth_1M_raw.ls']

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = MySentences('./data_util/ready_data/')

if args.type == "cbow":
    model = gensim.models.Word2Vec(sentences, min_count=5, size=300, workers=3, sg=0)
elif args.type == "skipgram":
    model = gensim.models.Word2Vec(sentences, min_count=5, size=300, workers=3, sg=1)

from gensim.models import word2vec
print( word2vec.FAST_VERSION )

from tempfile import mkstemp

model.save(args.out)
