# python3
import pandas as pd
from pprint import pprint
import gensim, logging, re, json, base64, traceback, sys
import smart_open, os, argparse
from logging.handlers import RotatingFileHandler

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


from tempfile import mkstemp

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering

parser = argparse.ArgumentParser(description='Load a gensim model and query it.')
parser.add_argument('-m', '--model', metavar='MODEL', type=str, help='model to load')
parser.add_argument('-f', '--format', metavar='FORMAT', type=str, help='format of the model: "w2v" or "gensim"')
parser.add_argument('-a', '--algo', metavar='ALGO', type=str, help='Clustering algorithm: "kmeans" or "spectral"')
parser.add_argument('-e', '--emojisize', metavar='EMOJISIZE', type=str, help='Emoji size in tsne (example: 4 or 0.1)')
parser.add_argument('-o', '--out', metavar='OUT', type=str, help='outputfile name (example: "test")')
args = parser.parse_args()

from gensim.models import Word2Vec, word2vec
print( word2vec.FAST_VERSION )

unicode_chart = "data_util/emojichart_v5.0.tsv"
faceemojis_path = "data_util/df_emoji_faces_std.tsv"

class log_wrap():
	def __init__(self):
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		self.logger = logging.getLogger()
		self.logger.setLevel(logging.DEBUG)
		formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
		file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(formatter)
		self.logger.addHandler(file_handler)
		steam_handler = logging.StreamHandler()
		steam_handler.setLevel(logging.DEBUG)
		self.logger.addHandler(steam_handler)
		self.logger.info('Logger initialized')

class count():
    def __init__(self, total=0):
        self.sc = 0
        self.total = total
    def plus(self):
        self.sc = self.sc + 1
        self.progress()
    def getsc(self):
        return self.sc
    def reset(self):
        self.sc = 0
        print("\n")
    def set_total(self, total):
        self.total = total
    def progress(self):
        self.progressBar(self.sc, self.total)
    def progressBar(self, value, endvalue, bar_length=100):
        '''print the progress bar given the values.
        default bar_length = 100'''
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

from itertools import islice

log = log_wrap()

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


if args.format == "gensim":
	model = gensim.models.Word2Vec.load(args.model) # open the model in gensim format
else:
	model = gensim.models.KeyedVectors.load_word2vec_format(args.model, binary=False, encoding="utf-8") # open the model in original format

df_emojichart = pd.read_csv(unicode_chart, sep='\t')
natives = [native for native in df_emojichart.Browser]

df_emojifaces = pd.read_csv(faceemojis_path, sep='\t')
faces = [face for face in df_emojifaces.Browser]


def native2imagename(native):
	name =  regex1.sub("", str(native.encode('unicode-escape'), 'utf-8'))
	name =  regex2.sub("-", name )
	name =  regex3.sub("-", name )
	name =  regex4.sub("", name )
	name = name.replace('--', '-')
	return name

n = 0
reduced = dict()

regex1 = re.compile(r"^\\U000", re.IGNORECASE)
regex2 = re.compile(r"\\U000", re.IGNORECASE)
regex3 = re.compile(r"\\u200d", re.IGNORECASE)
regex4 = re.compile(r"\\u", re.IGNORECASE)
for k in model.wv.vocab.keys():
	if k in natives:
		reduced[k] = model.wv.vocab[k]

def code2base64(code):
    encoded_string = ""
    with open("data_util/emojione_32px/"+ code +".png", "rb") as image_file:
    	encoded_string = 'data:image/png;base64,' +  base64.b64encode(image_file.read()).decode()
    return encoded_string


def imagename2targetword(imagename):
	target_word = ""
	try:
		target_word = code2base64(imagename)
	except:
		try:
			raise TypeError("Again ?!!")
		except:
			pass
		traceback.print_exc()
	return target_word

def tsne2layout(model):
	log.logger.info("Starting graph")
	log.logger.info("X creation")
	X = model[reduced]
	log.logger.info("Init TSNE and fit_transform")
	sizeemoji = args.emojisize # we used 4 or 0.1
	if float(sizeemoji) > 1 :
		tsne = TSNE(n_components=2, verbose=1, perplexity=30, learning_rate=100.0, early_exaggeration=2.0)
	else:
		tsne = TSNE(n_components=2, verbose=1, perplexity=50, learning_rate=400.0, early_exaggeration=2.0) 

	X_tsne = tsne.fit_transform(X)
	log.logger.info("Scattering TSNE")
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

	rows = list(reduced.keys())

	ensemble = list()
	jsdata = dict()
	labels = list()
	data = list()
	l_x = list()
	l_y = list()
	l_text = list()
	images = list()
	point = {'x':[], 'y':[], 'mode':'markers+text', 'name':'Team B', 'text':[], 'marker':{'size':8}}

	for row_id in range(0, len(reduced.keys())):
		target_word = rows[row_id]
		if target_word not in faces : continue
		target_word = native2imagename(target_word)

		x = X_tsne[row_id, 0].item()
		y = X_tsne[row_id, 1].item()

		try:
			target_word = code2base64(target_word)
		except:
			try:
				raise TypeError("Again ?!!")
			except:
				pass
			traceback.print_exc()

		ensemble.append([target_word, x, y])
		plt.annotate(target_word, (x,y))
		labels.append(target_word)
		images.append({'source':target_word, 'xref':'x', 'yref':'y', 'x':x, 'y':y, "sizex": sizeemoji,"sizey": sizeemoji, "xanchor": "left", "yanchor": "bottom"})

	layout = { 'xaxis': { 'range': [ -30, 30 ] }, 'yaxis': { 'range': [-30, 30] }, 'title':'Data Labels Hover', 'images': images, 'autosize':True, 'dragmode': "pan", 'autoexpand': True, 'width':1300, 'height':600}
	return layout

def createhtml(layout):
	html = """<!doctype html>
	<html lang="en">
	<head>
	<link rel="stylesheet" href="//cdnjs.cloudflare.com/ajax/libs/emojify.js/1.1.0/css/basic/emojify.min.css" />
	<!-- Plotly.js -->
	<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
	</head>
	<body>
	<div id="myDiv" style="width:100%; height:100%;"><!-- Plotly chart will be drawn inside this DIV --></div>

	<!--<img id="jpg-export"></img>-->
	<script src="//cdnjs.cloudflare.com/ajax/libs/emojify.js/1.1.0/js/emojify.min.js"></script>
	<script>
	var d3 = Plotly.d3;
	var img_jpg= d3.select('#jpg-export');
	//var data = DATA_HERE;
	var layout = LAYOUT_HERE;
	//Plotly.newPlot('myDiv', data, layout, {scrollZoom: true});
	Plotly.newPlot('myDiv', [], layout, {scrollZoom: true});
	/*.then(
		function(gd)
		{
		Plotly.toImage(gd,{height:1300,width:1300})
			.then(
				function(url)
			{
				img_jpg.attr("src", url);
				return Plotly.toImage(gd,{format:'png',height:1300,width:1300});
			}
			)
		});*/
	</script>
	</body>
	</html>"""
	html = html.replace('LAYOUT_HERE', json.dumps(layout))
	html_output = open("tsne_" + args.out +".html","w") 
	html_output.write(html)
	html_output.close() 


def clustering_emojis():
	X = model[reduced]
	n_clusters = 63 #our number of emojis
	print( "n_clusters", n_clusters  )

	if args.algo == "kmeans":
		clustering_model = KMeans(n_clusters=n_clusters, random_state=1, n_init=1000, n_jobs=4, max_iter=500, verbose=0 ).fit_predict(X)
	elif args.algo == "spectral":
		clustering_model = SpectralClustering(n_clusters=n_clusters, eigen_solver=None, random_state=1, n_init=4000, gamma=0.7, affinity='rbf', n_neighbors=2, eigen_tol=0.0, assign_labels='discretize', degree=3, coef0=1, kernel_params=None, n_jobs=-1).fit_predict(X)

	rows = list(reduced.keys())

	img_tags = list()
	append_img_tags = img_tags.append

	d_divs = dict()
	for i in range(0, n_clusters):
		d_divs[i] = []

	for row_id in range(0, len(reduced.keys())):
		if rows[row_id] not in faces : continue
		d_divs[clustering_model[row_id]].append(  rows[row_id]   )
	
	image_clusters = list()
	append_image_clusters = image_clusters.append

	for key, value in d_divs.items():
		img_tags = list()
		append_img_tags = img_tags.append
		for v in value:
			image_name = native2imagename(v)
			target_word = imagename2targetword(image_name)
			append_img_tags(   "<img src='{}' height='42' width='42'>".format(target_word)  )
		append_image_clusters( img_tags )
	
	divs = list()
	append_divs = divs.append
	for image_cluster in image_clusters :
		if len(image_cluster) == 0: continue
		div = ["<div class='col-md-6'><div class='panel panel-default'><div class='panel-body'><div class='cluster'>"] + image_cluster + ["</div></div></div></div>"]
		append_divs( "".join(div) )

	print( len(divs), 'clusters', len(faces), 'faces' )
	return divs

def get_clusters():
	divs = clustering_emojis()

	html = """<!DOCTYPE html>
		<html lang="en">
		<head>
		<title>Bootstrap Example</title>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
		<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.0/jquery.min.js"></script>
		<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
		</head>
		<body>

		<div class="container">
		<br/>
		<div class="row">
		INSERT_DIVS_HERE
		</div>
		</div>

		</body>
		</html>"""
	html = html.replace("INSERT_DIVS_HERE", "".join(divs))
	html_output = open("clusters_" + args.out+ ".html","w") 
	html_output.write(html)
	html_output.close()


def get_tsne():
	layout = tsne2layout(model)
	createhtml(layout)

get_tsne()
get_clusters()