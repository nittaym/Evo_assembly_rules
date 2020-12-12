'''
Created on Jul 29, 2015

@author: yonatanf
'''


import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from matplotlib import cm
from numpy import linspace, sqrt, array, zeros, arange
# from GoreUtilities.graph import array2colors
from itertools import combinations
from matplotlib import patches as mpatches


def array2colors(x, cmap=cm.jet, **kwargs):
	'''
	Return rgba colors corresponding to values of x from desired colormap.
	Inputs:
		x		 = 1D iterable of strs/floats/ints to be mapped to colors.
		cmap	  = either color map instance or name of colormap as string. 
		vmin/vmax = (optional) float/int min/max values for the mapping.
					If not provided, set to the min/max of x.
	Outputs:
		colors = array of rgba color values. each row corresponds to a value in x.
	'''
	from matplotlib.colors import rgb2hex
	## get the colormap
	if type(cmap) is str: 
		if cmap not in cm.datad: raise ValueError('Unkown colormap %s' %cmap)
		cmap = cm.get_cmap(cmap)
	
	x = np.asarray(x)
	isstr = np.issubdtype(x.dtype, str)
	if isstr:
		temp = np.copy(x)
		x_set = set(x)
		temp_d = dict((val,i) for i,val in enumerate(x_set))
		x = [temp_d[id] for id in temp]   
	## get the color limits
	vmin = kwargs.get('vmin', np.min(x))
	vmax = kwargs.get('vmax', np.max(x))
	## set the mapping object
	t = cm.ScalarMappable(cmap=cmap)
	t.set_clim(vmin, vmax)
	## get the colors
	colors = t.to_rgba(x)
	if hex:
	   colors = [rgb2hex(c) for c in colors] 
	return colors


def embedded_pie(compostion, location, size, colors, ax=None):
	if ax is None:
		fig, ax = subplots()
	compostion = [0] + list(np.cumsum(compostion)* 1.0 / sum(compostion))
	n = len(compostion)
	for i in xrange(n-1):
		c = colors[i]
#		 theta1 = compostion[i]*360
#		 theta2 = compostion[i+1]*360
#		 w = Wedge(location, size, theta1, theta2, fc=c, ec='k', lw=0, clip_on=False)
#		 ax.add_patch(w)
		x = [0] + np.cos(np.linspace(2 * np.pi * compostion[i], 2 * np.pi * compostion[i+1], 20)).tolist()
		y = [0] + np.sin(np.linspace(2 * np.pi * compostion[i], 2 * np.pi * compostion[i+1], 20)).tolist()
		xy = zip(x,y)
		ax.scatter( location[0], location[1], marker=(xy,0), s=size, facecolor=c, clip_on=False, lw=0.1, c='w')
	

def outcome_dict2df(outcome):
	'''
	convert a dictionary keyed by species pairs to dataframe
	'''
	sps = np.unique(outcome.keys())
	df = pd.DataFrame(columns=sps, index=sps)
	for (s1,s2), o in outcome.iteritems():
		df.loc[s1,s2] = o
		df.loc[s2,s1] = o
	return df

def make_interaction_network(outcomes, pair_fracs=None, directed=False):
	if directed:
		net = nx.DiGraph()
	else:
		net = nx.Graph()
	if not hasattr(outcomes, 'columns'):
		outcomes = outcome_dict2df(outcomes)
	net.add_nodes_from(outcomes.columns)
	for s1,s2 in combinations(net.nodes(), 2):
		o = outcomes.loc[s1,s2]
		if o in net.nodes():
			winner = o
			loser  = s1 if s1!=o else s2
			net.add_edge(loser, winner, outcome=o,)
		elif o=='COX':
			if pair_fracs is None:
				frac = .6
			elif isinstance(pair_fracs, float):
				frac = pair_fracs
			else:
				f1 = pair_fracs.ix[s1,s2,s1]
				f2 = pair_fracs.ix[s2,s1,s1]
				frac = pd.Series([f1,f2]).mean()
			net.add_edge(s1, s2, outcome='coexistence', fraction=frac)
		elif o=='BIS': 
			net.add_edge(s1, s2, outcome='bistability',)
		else:
			net.add_edge(s1, s2, outcome='NA',)
	return net


class NetPloter(object):
		
	def __init__(self, ax=None, pos=None, scale=1, species_colors=None, net=None, arrow_opt=dict(), style='old', order=None):
		self.ax   = ax
		self.scale = scale
		self.species_colors = species_colors
		self.net = net
		
		if pos is None:
			self.verts = self.get_vertices(order)
		else:
			self.verts = pos
		
		opt = {'head_width': 0.04, 'head_length': 0.08, 'width': 0.01,
				'length_includes_head': True, 'ec':'w', 'linewidth':.1}
		
		for k,v in opt.items():
			arrow_opt.setdefault(k,v)
	
		if style is 'old':
			self.boundry_kwargs = {'dominance': dict(color='DarkRed', **arrow_opt),
								   'coexistence': dict(color='DarkBlue', **arrow_opt),
								   'bistability': dict(color='k', **arrow_opt),
								   'NA': dict(color='gray', linewidth=1),
								   }
		elif style is 'new':
			self.boundry_kwargs = {'dominance': dict(color='#e41a1c', **arrow_opt),
						'coexistence': dict(color='#377eb8', **arrow_opt),
						'bistability': dict(color='#4daf4a', **arrow_opt),
						'NA': dict(color='gray', linewidth=1),
						}

		elif style is 'exc_only':
			self.boundry_kwargs = {'dominance': dict(color='#e41a1c', **arrow_opt),
						'coexistence': dict(color='w', **arrow_opt),
						'bistability': dict(color='#4daf4a', **arrow_opt),
						'NA': dict(color='gray', linewidth=1),
						}
		
	def get_vertices(self, order):
		pos = pd.DataFrame(nx.drawing.circular_layout(self.net)).T
		pos.columns=['x','y']
		if order is not None:
			#print pos.index
			nodes = list(self.net.nodes())
			pos.index = [order[nodes.index(n)] for n in pos.index]
		return pos
	
	
	def _remove_axis(self, e=.2):
		ax = self.ax
		lim = (-e, self.scale+e)
		ax.set_xlim(lim)
		ax.set_ylim(lim)
		ax.axis('off')
	
	
	def _plot_arrows(self, points, fs=None, n=4 ,**kwargs):
		'''
		points : DataFrame (2x2)
			First row is origin and second is target.
			Columns are x,y values.
		'''
		ax = self.ax
		if fs is None:
			fs = linspace(0,1,n+1)[:-1]
		origin = points.loc['origin']
		target = points.loc['target']
		for i,f in enumerate(fs):
			target_x = f*points.loc['origin','x'] + (1-f)*points.loc['target','x']
			target_y = f*points.loc['origin','y'] + (1-f)*points.loc['target','y']
			ax.arrow(origin['x'], origin['y'], target_x-origin['x'], target_y-origin['y'], **kwargs)
	
	def _make_points(self, origin, target, offsets=zeros(2)):
		points = pd.DataFrame([origin, target], index=['origin', 'target'])

		dx, dy = points.diff().iloc[-1]
		theta = np.arctan(dy/dx)
		ey = abs(np.sin(theta)*offsets)
		ex = abs(np.cos(theta)*offsets)
		snx = np.sign(dx)
		sny = np.sign(dy)

		points_offset = points.copy()

		points_offset.iloc[0,0] += snx*ex[0]
		points_offset.iloc[1,0] -= snx*ex[1]
		points_offset.iloc[0,1] += sny*ey[0]
		points_offset.iloc[1,1] -= sny*ey[1]
		return points_offset
	
	def plot_net_boundry(self, label_vertices=True, offset=.1, vert_defaults={}, text_kwargs={}):
		if self.ax is None:
			fig, ax = subplots(figsize=(6,6))
			self.ax = ax
		else:
			ax = self.ax
		net = self.net
		verts = self.verts
		labels = verts.index
		for s1,s2 in net.edges():
			edge = net[s1][s2]
			o =  edge['outcome']
			if o in labels: #if dominance
				origin = s2 if o==s1 else s1
				target = o
				n = 2
				offsets = array([offset]*2)
				points = self._make_points(verts.loc[origin], verts.loc[target], offsets=offsets)
				self._plot_arrows(points, n=n, **self.boundry_kwargs['dominance'])
			elif o=='bistability':
				middle = verts.loc[[s1,s2]].mean()
				n = 2
				offsets = array([0, offset])
				points = self._make_points(middle, verts.loc[s1], offsets=offsets)
				self._plot_arrows(points, n=n, **self.boundry_kwargs[o])
				points = self._make_points(middle, verts.loc[s2], offsets=offsets)
				self._plot_arrows(points, n=n, **self.boundry_kwargs[o])
			elif o=='coexistence':
		#continue
				offsets = array([offset]*2)
				points = self._make_points(verts.loc[s1], verts.loc[s2], offsets=offsets)
				f = edge['fraction']
				fixed_point = f*points.iloc[0] + (1-f)*points.iloc[1]
				n = 1
				offsets = array([offset,0.01])
				points = self._make_points(verts.loc[s1], fixed_point, offsets=offsets)
				self._plot_arrows(points, n=n, **self.boundry_kwargs[o])
				points = self._make_points(verts.loc[s2], fixed_point, offsets=offsets)
				self._plot_arrows(points, n=n, **self.boundry_kwargs[o])
				ax.plot(fixed_point['x'], fixed_point['y'], '*k', ms=10, c=self.boundry_kwargs[o]['color'], mew=.5)
			else:
				ax.plot(verts.loc[(s1,s2), 'x'], verts.loc[(s1,s2), 'y'], **self.boundry_kwargs[o])
		if label_vertices:
			self.label_vertices(labels, vert_defaults, text_kwargs)
		self._remove_axis()
		
	def label_vertices(self, labels, default_props={}, text_kwargs={}):
		n = len(labels)
		verts = self.verts
		ax = self.ax
#		 if self.species_colors is not None:
#			 bboxes=[dict(facecolor=self.species_colors[l], alpha=0.7, lw=0.1, boxstyle='round,pad=.1') for l in labels]
#		 else:
#			 bboxes = [None]*n
		bboxes = [None]*n
		text_kwargs_defaults = dict(va='center', ha='center', color='k', size='xx-large')
		text_kwargs_defaults.update(text_kwargs)
		for i in range(n):
			props = dict()
			if isinstance(self.species_colors, dict):
				props.setdefault('facecolor', self.species_colors[labels[i]])
			else:
				props.setdefault('facecolor', self.species_colors)
			props.update(default_props)
			props.setdefault('edgecolor', 'k')
			props.setdefault('linewidth', 2	)
			props.setdefault('radius', .1)
			ax.text(verts.iloc[i,0], verts.iloc[i,1], labels[i], bbox=bboxes[i], **text_kwargs_defaults)
			rad = props.pop('radius')
			patch = mpatches.Circle(verts.iloc[i].values, radius=rad, **props)
			ax.add_artist(patch)


	
class TernPloter(object):
	
	def __init__(self, ax=None, scale=1, species_colors=None, net=None, style='old'):
		self.ax   = ax
		self.scale = scale
		self.species_colors = species_colors
		self.net = net
		
		opt = {'head_width': 0.04, 'head_length': 0.08, 'width': 0.01,
				'length_includes_head': True, 'ec':'w', 'linewidth':.1, 'alpha':.9}

		self.arrow_kwargs_default = opt

		if style is 'old':
			self.boundry_kwargs = {'dominance': dict(color='DarkRed', **opt),
								   'coexistence': dict(color='DarkBlue', **opt),
								   'bistability': dict(color='k', **opt),
								   'NA': dict(color='gray', linewidth=1),
								   }
		elif style is 'new':
			self.boundry_kwargs = {'dominance': dict(color='#e41a1c', **opt),
						'coexistence': dict(color='#377eb8', **opt),
						'bistability': dict(color='#4daf4a', **opt),
						'NA': dict(color='gray', linewidth=1),
						}
		
		
	def get_barycentric_coords(self, frame):
		'''
		'''
		if frame.shape[1] !=3:
			raise ValueError('frame must have exactly 3 columns.')
		try:
			vals = frame.values
		except AttributeError:
			vals = frame

		scale = self.scale
		(a,b,c) = [vals[:,i] for i in range(3)]
		x = 0.5 * ( 2.*b+c ) / ( a+b+c )
		y = 0.5*sqrt(3) * c / (a+b+c)
		xy = scale*array([x,y]).T
		return pd.DataFrame(xy, columns=['x', 'y'])

	def get_vertices(self):
		vals = array([[1,0,0],
					  [0,1,0],
					  [0,0, 1]])
		return self.get_barycentric_coords(vals)
	
	
	def _remove_axis(self, e=.1):
		ax = self.ax
		lim = (-e, self.scale+e)
		ax.set_xlim(lim)
		ax.set_ylim(lim)
		ax.axis('off')
	
	def plot_boundry(self, labels=None,**kwargs):
		if self.ax is None:
			fig, ax = subplots(figsize=(6,6))
			self.ax = ax
		ax =self.ax
		verts = self.get_vertices()
		verts.loc[4] = verts.loc[0]
		kwargs.setdefault('lw',2)
		kwargs.setdefault('c','k')
		boundry = ax.plot(verts['x'], verts['y'], **kwargs)
		self._remove_axis()
		if labels is not None:
			self.label_vertices(labels)

		return boundry
	
#	 def _plot_arrows(self, origin, target, fs=None, n=4, **kwargs):
#		 ax = self.ax
#		 if fs is None:
#			 fs = linspace(0,1,n+1)
#		 for i,f in enumerate(fs):
#			 target_x = f*origin.loc['x'] + (1-f)*target.loc['x']
#			 target_y = f*origin.loc['y'] + (1-f)*target.loc['y']
#			 ax.annotate( '', 
#				 xy=(target_x,target_y), xycoords='data',		#target point
#				 xytext=origin, textcoords='data', #origin point
#				 arrowprops=dict( **kwargs)
# #				 arrowprops=dict(arrowstyle="->", **kwargs)
#				 )

	def _plot_arrows(self, origin, target, fs=None, n=4, **kwargs):
		ax = self.ax
		if fs is None:
			fs = linspace(0,1,n+1)[:-1]
		for i,f in enumerate(fs):
			target_x = f*origin.loc['x'] + (1-f)*target.loc['x']
			target_y = f*origin.loc['y'] + (1-f)*target.loc['y']	
			ax.arrow(origin['x'], origin['y'], target_x-origin['x'], target_y-origin['y'], **kwargs)
	
	def plot_net_boundry(self, labels, label_vertices=False, labels_kwargs=dict()):
		if self.net is None:
			return self.plot_boundry(labels)
		else:
			net = self.net
		if self.ax is None:
			fig, ax = subplots(figsize=(6,6))
			self.ax = ax
		else:
			ax = self.ax
		verts = self.get_vertices()
		verts.index = labels
		for s1,s2 in net.edges():
			edge = net[s1][s2]
			o =  edge['outcome']
			if o in labels: #if dominance
				origin = s2 if o==s1 else s1
				target = o
				n = 2
				self._plot_arrows(verts.loc[origin], verts.loc[target], n=n, **self.boundry_kwargs['dominance'])
			elif o=='bistability':
				middle = verts.loc[[s1,s2]].mean()
				n = 3
				self._plot_arrows(middle, verts.loc[s1], n=n, **self.boundry_kwargs[o])
				self._plot_arrows(middle, verts.loc[s2], n=n, **self.boundry_kwargs[o])
			elif o=='coexistence':
				t = pd.DataFrame(zeros((1,3)), columns=labels)
				t[s1] = edge['fraction']
				t[s2] = 1-edge['fraction']
				fixed_point= self.get_barycentric_coords(t).loc[0]
				n = 1

				margin = .05
				f1 = (1+margin)*edge['fraction']
				t1 = t.copy()
				t1[s1] = f1
				t1[s2] = 1-f1
				fp1= self.get_barycentric_coords(t1).loc[0]

				f2 = (1-margin)*edge['fraction']
				t2 = t.copy()
				t2[s1] = f2
				t2[s2] = 1-f2
				fp2= self.get_barycentric_coords(t2).loc[0]

				# self._plot_arrows(verts.loc[s1], fp1, n=n, **self.boundry_kwargs[o])
				# self._plot_arrows(verts.loc[s2], fp2, n=n, **self.boundry_kwargs[o])
				ax.plot(fixed_point['x'], fixed_point['y'], '*k', ms=20, c='k', mew=.5)
			else:
				ax.plot(verts.loc[(s1,s2), 'x'], verts.loc[(s1,s2), 'y'], **self.boundry_kwargs[o])
		if label_vertices:
			self.label_vertices(labels, **labels_kwargs)
		self._remove_axis()
		
	def label_vertices(self, labels, **kwargs):
		verts = self.get_vertices()
		ax = self.ax
		if self.species_colors is not None:
			bboxes=[dict(facecolor=self.species_colors[l], alpha=0.7, lw=0, boxstyle='round,pad=.1') for l in labels]
		else:
			bboxes = [None]*3
		vas = ['top', 'top', 'bottom']
		has = ['right', 'left', 'center']
		kwargs.setdefault('color','k')
		kwargs.setdefault('size', 'x-large')
		for i in range(3):
			ax.text(verts.iloc[i,0], verts.iloc[i,1], labels[i], va=vas[i], ha=has[i], 
					bbox=bboxes[i], **kwargs)
	
	def plot(self, frame, plot_func='scatter', fs=[.3,.7], labels_kwargs=dict(), plot_boundry=True, **kwargs):
		label_vertices = kwargs.pop('label_vertices', True)
		if plot_boundry:
			self.plot_net_boundry(frame.columns, label_vertices=label_vertices, labels_kwargs=labels_kwargs)
		xy = self.get_barycentric_coords(frame)
		if plot_func=='arrow_path':
			arrow_kwargs = kwargs.copy()
			for k,v in self.arrow_kwargs_default.items():
				if k not in arrow_kwargs:
					arrow_kwargs[k] = v
			n = xy.shape[0]
			out = []
			for i in range(n-1):
				o = self._plot_arrows(xy.iloc[i], xy.iloc[i+1], fs=fs , **arrow_kwargs)
				out.append(o)
		else:
			func = getattr(self.ax, plot_func)
			kwargs.setdefault('zorder', 10)
			out = func(xy['x'], xy['y'], **kwargs)
		return out

if __name__ == '__main__':		
	a = array([[1,0,0],
			   [0,1,0],
			   [0,0,1],
			   [.5,.25,.25,],
			   [.25,.5,.25,],
			   [.25,.25,.5,],
			   [.24,.24,.52,],
			   [.5,.5,0],
			   [.5,0,.5],
			   [0,.5,.5],
			   ])
	
	net = nx.Graph()
	# net.add_nodes_from(tmp.columns)
	net.add_edge('Ea', 'Pa', outcome='Pa',)
	net.add_edge('Ea', 'Pch', outcome='bistability',)
	net.add_edge('Pch', 'Pa', outcome='coexistence', fraction=.5)
	
	frame = pd.DataFrame(a, columns=['Ea','Pch','Pa'])
	
	species = [ 'Ea', 'Pa', 'Pch', 'Pci', 'Pf', 'Pp' , 'Pv', 'Sm']
	species_colors = array2colors(arange(8), cmap=cm.Accent)
	species_colors_d = {species[i]:species_colors[i] for i in range(8) }
	
	fig, ax = subplots()
	tp = TernPloter(scale=1, species_colors=species_colors_d, net=net, ax=ax)
	# tp.plot_net_boundry(['Ea','Pch','Pa'], label_vertices=False)
	labels_kwargs = {'size':22}
	tp.plot(frame, marker='o', s=15**2, alpha=.7, c=array2colors(arange(a.shape[0]), cmap=cm.jet), 
			labels_kwargs=labels_kwargs)
	
	
	
	# fig, ax = subplots(figsize=(10,5))
	# compostion = np.random.rand(20)
	# location = [1,1]
	# compostion = [0] + list(np.cumsum(compostion)* 1.0 / sum(compostion))
	# n = len(compostion)
	# colors = array2colors(arange(n))
	# embedded_pie(compostion, location, 400**2, colors, ax=ax)
	# ax.set_xlim(0,2)
	# ax.set_ylim(0,2)



plt.show()
