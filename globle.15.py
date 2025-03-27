#!/usr/bin/env python
# coding: utf8
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
from itertools import groupby
from matplotlib import cm
from pathos.multiprocessing import ProcessingPool as Pool
from scipy import signal
from scipy.sparse import csr_matrix as sparse
import arrow
import bisect
import copy 
import csv
import datetime
import difflib
import glob
import math
import matplotlib.pyplot as plt
import mdptoolbox,mdptoolbox.example
import multiprocessing
import numpy as np
import operator
import os
import random
import socket
import sys
import time
import psutil
from appJar import gui
TT=str(int(time.time()*1000))
Directory=1
A=1
b=1
M=1
d=1
cN=1
cM=1
cOn=1
h=1
w=1
real_arrival=1
histogram=1
c1,t1,m1=1,1,1
c2,t2,m2=1,1,1
c3,t3,m3=1,1,1
c4,t4,m4=1,1,1
c5,t5,m5=1,1,1
png1=1
m0=1
################################################################################
def print1(x):
	print x
	app.addListItems('log',[x])
################################################################################
def latex_arrivee(real_arrival,A,b,M,d,cN,cM,cOn,h,w,c1,c2,c3,c4,c5,t1,t2,t3,t4,t5,m1,m2,m3,m4,m5,png1):
	begin_latex=r"""
	\documentclass[a4paper,12pt]{article}
	\usepackage[utf8]{inputenc}
	\usepackage[T1]{fontenc}
	\usepackage[french]{babel}
	\usepackage[mediumspace,mediumqspace,Grey,squaren]{SIunits}
	\usepackage[french,lined,ruled,onelanguage,commentsnumbered,linesnumbered,inoutnumbered]{algorithm2e}
	\SetEndCharOfAlgoLine{}
	%\usepackage{algorithmic}
	\usepackage{amsfonts}
	\usepackage{amsmath,bm,times}
	\usepackage{amssymb}
	\usepackage{amstext}
	\usepackage{amsthm}
	\usepackage{array}
	\usepackage{bibunits}
	\usepackage{blkarray}
	\usepackage{booktabs}
	\usepackage{breakcites}
	\usepackage{calc}
	\usepackage{changepage}
	\usepackage{color}
	\usepackage{dsfont}
	%\usepackage{enumitem}
	\usepackage{epsfig}
	\usepackage{eurosym}
	\usepackage{fancyhdr}
	\usepackage{float}
	\usepackage{graphics}
	\usepackage{graphicx}
	\usepackage{hyperref}
	\usepackage{lettrine}
	\usepackage{listings}
	\usepackage{makecell}
	%\usepackage{makeidx}
	\usepackage{mathrsfs}
	\usepackage{microtype} %\cite spills over into the margin,latex how break citation end of line
	\usepackage{multicol}
	\usepackage{nccmath}
	\usepackage{paralist}
	\usepackage{pdflscape}
	\usepackage{pgfplotstable}
	\usepackage{pgfplots}
	\usepackage{pifont}
	\usepackage{pslatex}
	\usepackage{rotating}
	\usepackage{subfigure}
	\usepackage{tikz}
	\usepackage{verbatim}
	\usepackage{a4wide}
	\usepackage{tasks}
	\usepackage{tabularx}
	\usetikzlibrary{calc,shadings}
	\usetikzlibrary{chains,shapes.multipart}
	\usetikzlibrary{shapes,arrows}
	\usetikzlibrary{shapes,calc}
	\usetikzlibrary{snakes,arrows,shapes}
	\theoremstyle{definition}\newtheorem{exemple}{Exemple}[section]
	\theoremstyle{definition}\newtheorem{exercice}{Exercice}%[subsection]
	\theoremstyle{definition}\newtheorem{notion}{Notion}[section]
	\usepackage{listings}
	\definecolor{javared}{rgb}{0.6,0,0} % for strings
	\definecolor{javagreen}{rgb}{0.25,0.5,0.35} % comments
	\definecolor{javapurple}{rgb}{0.5,0,0.35} % keywords
	\definecolor{javadocblue}{rgb}{0.25,0.35,0.75} % javadoc
	\definecolor{deepblue}{rgb}{0,0,0.5}
	\definecolor{deepred}{rgb}{0.6,0,0}
	\definecolor{deepgreen}{rgb}{0,0.5,0}
	\definecolor{javared}{rgb}{0.6,0,0} % for strings
	\definecolor{javagreen}{rgb}{0.25,0.5,0.35} % comments
	\definecolor{javapurple}{rgb}{0.5,0,0.35} % keywords
	\definecolor{javadocblue}{rgb}{0.25,0.35,0.75} % javadoc
	\definecolor{orange}{HTML}{FF7F00}
	\setlength\parindent{0pt}
	%----------------------------------------------------------------------------------------    
	\lstset{
	    language=java,
	    basicstyle=\footnotesize\ttfamily\bfseries,
	    otherkeywords={keolis},
	    tabsize=3,
	    %frame=lines,
	    %caption=Test,
	    label=code:sample,
	    frame=shadowbox,
	    rulesepcolor=\color{gray},
	    xleftmargin=20pt,
	    framexleftmargin=15pt,
	    keywordstyle=\color{blue},
	    commentstyle=\color{OliveGreen},
	    stringstyle=\color{red},
	    numbers=left,
	    numberstyle=\color{deepblue},
	    numbersep=5pt,
	    breaklines=true,
	    showstringspaces=false,
	    emph={ts_to_local_time_str,local_time_str_to_ts,detect_delimiter,get_closest_index,dot_group_0,imsi_code_tac_to_list,code_tac_MtoM_to_set,ss_csv_to_list,all_zones_to_dict,national_zone_csv_to_dict,etranger_zone_csv_to_dict,trim_ss,ss_to_alias_list,aggregate_same_zone,get_excursions,duree_pi,duree_nu,longest_excursion,all_trips,all_trips_r3,longest_stop,get_first_3h00,get_restitution_periodes,redressement_masking,print_indicateur,indicateur_to_csv,adjust_residents,r1,r2,r3,r4,filter_MtoM,indicators1234,study_quantum},
	    emphstyle=\color{deepgreen}
	    }
	%----------------------------------------------------------------------------------------    
	%opening
	\definecolor{histoblue}{HTML}{3090C7}
	%\definecolor{histoblue}{HTML}{1F45FC}
	\definecolor{histored}{HTML}{F70D1A}
	%\definecolor{histored}{HTML}{red!10}
	\usetikzlibrary{calc,shadings}
	\usetikzlibrary{chains,shapes.multipart}
	\usetikzlibrary{shapes,calc}
	\definecolor{dkgreen}{rgb}{0,0.6,0}
	\definecolor{gray}{rgb}{0.5,0.5,0.5}
	\usepackage[local,copyexercisesinsolutions]{exsol}
	\pagestyle{plain}

	\usepackage{amsmath}
	\newcommand{\St}{{{\cal{S}}}}
	\newcommand{\A}{{{\cal{A}}}}
	\newcommand {\Prr}{{{\cal{P}}}}
	\newcommand{\Ct}{{{\cal{C}}}}
	\newcommand{\HH}{{{\cal{H}}}}
	\newcommand{\LL}{{{\cal{L}}}}
	\newcommand{\G}{{{\cal{G}}}}
	\newcommand{\OO}{{{\mathcal{O}}}}
	\newcommand{\maxx}{{{\mathcal{M}}}}
	%\mathcal{O}

	\newcommand{\eeuro}{} 
	\renewcommand{\euro}{} 
	\newcommand{\PP}{{{\bfcal{P}}}}
	\newcommand{\E}{\mathbb{E}} 
	\newcommand{\Var}{\mathrm{Var}}
	\newcommand{\Cov}{\mathrm{Cov}}
	\newcommand{\st}{\leq_{st}}

	%opening
	\title{P2-Algorithmique et Complexite\\Cours 1\\Les Algorithmes}
	\author{Lea}


	%------------------------------------------------------------------------    
	%------------------------------------------------------------------------    
	%------------------------------------------------------------------------    
	%------------------------------------------------------------------------    
	%------------------------------------------------------------------------    
	%\newcommand{\exo1}{}
	%------------------------------------------------------------------------    
	%\newcommand{\myalgorithm}{}
	%------------------------------------------------------------------------    
	%------------------------------------------------------------------------    
	%------------------------------------------------------------------------    
	%------------------------------------------------------------------------    


	\begin{document}"""

	c1,c2,c3,c4,c5,t1,t2,t3,t4,t5,m1,m2,m3,m4,m5=[int(i)for i in [c1,c2,c3,c4,c5,t1,t2,t3,t4,t5,m1,m2,m3,m4,m5]]
	mm=min(m1,m2,m3,m4,m5)*.9
	m1,m2,m3,m4,m5=[i-mm for i in [m1,m2,m3,m4,m5]]
	file=TT+'file'+'.tex'
	print1(('Latex source',file))
	# file='file123.tex'
	maxP=max(A.values())
	maxA=max(A.keys())
	with open(file,"w") as f:
		f.write(begin_latex+"\n")	
		bfigure=r"""
		%============================================================
		\begin{figure}[H]
		\centering
		\resizebox{\linewidth}{!}{
		\begin{tikzpicture}
		\begin{axis}[%
		%xmode=log,
		grid=both,
		width=6.0in,
		height=4.0in,
		%every axis plot/.append style={ultra thick},
		x tick label style={
		%rotate=45,
		 /pgf/number format/.cd,
		set thousands separator={},
			 fixed
		 },
		at={(0in,0in)},
		xlabel={Number of jobs},
		ylabel={\;\;\;\;Probability},
		scale only axis,
		separate axis lines,
		every outer x axis line/.append style={white!15!black},
		every x tick label/.append style={font=\color{white!15!black}},
		xmin=0,
		xmax="""+str(maxA)+r""",
		every outer y axis line/.append style={white!15!black},
		every y tick label/.append style={font=\color{white!15!black}},
		ymin=0,
		ymax="""+str(maxP*1.1)+r""",
		%legend style={draw=white!15!black,legend cell align=left,cells={align=left}}
		legend style={draw=white!15!black,legend cell align=left,legend pos=north east}
		]
		%\addplot [color=red,mark=triangle*]
		\addplot [color=red]
		table[row sep=crcr]{%"""
		lines=r""
		for k in A:lines+=str(k)+' '+str(A[k])+r'\\'+'\n'
		efigure=r"""		};
		\addlegendentry{Arrival histogram with $|S_A|="""+str(len(A))+r"""$ and $\max(S_A)="""+str(max(A))+r"""$.};

		\end{axis}
		\end{tikzpicture}%
		}
		\caption[Arrival traffic distribution]{Arrival traffic distribution.}
		\label{compa:"""+str(time.time())+"""}
		\end{figure}
		%============================================================
		"""
		f.write(bfigure+"\n")	
		f.write(lines)	
		f.write(efigure+"\n")	
		btable=r"""
		%============================================================
		\begin{table}[H]
		\caption{Settings of the numerical analysis.}
		\centering 
		%\resizebox{\columnwidth}{!}
		{
		\begin{tabularx}{\columnwidth}{cccX}\toprule
		%\begin{tabular}{cccl}\toprule
		\textbf{Parameters}&\textbf{Value}&\textbf{Unit}&\textbf{Description}\\\midrule
		$\maxx$&"""+str(M)+r"""&servers&total number of servers\\
		$d$&"""+str(d)+r"""&jobs/server&processing capacity of a server\\
		$b$&"""+str(b)+r"""&jobs&buffer size\\
		$h$&"""+str(h)+r"""&slots&horizon\\
		$w$&"""+str(w)+r"""&slots&window\\
		$|S_A|$&"""+str(len(A))+r"""&bins&arrival jobs histogram size\\
		$\max(S_A)$&"""+str(max(A))+r"""&jobs&max arrival jobs per slot\\
		$\E(\HH_A)$&"""+str(round(expectation(A),2))+r"""&jobs&expected arrival jobs per slot\\
		$c_M$&"""+str(cM)+r"""&\euro&cost of energy needed by a server\\
		$c_N$&"""+str(cN)+r"""&\euro&cost of waiting a job\\
		$c_{On}$&"""+str(cOn)+r"""&\euro&cost of switching on a server\\
		\bottomrule
		\end{tabularx}
		}
		\label{Greedy:tab:"""+str(time.time())+r"""}
		\end{table} 
		%============================================================
		"""
		f.write(btable+"\n")	
		page1=r"""
		%============================================================
		\begin{figure}[p]
		\centering
		\includegraphics[width=\textwidth]{"""+png1+r"""}
		\caption[Threshold results]{Threshold results.}
		\label{Threshold:4:"""+str(time.time())+r"""}
		\end{figure}
		%============================================================
		"""
		f.write(page1+"\n")	

		max1=max(c1,c2,c3,c4,c5)*1.2
		page2=r"""
		%============================================================
		\begin{figure}[H]
		\centering
		\resizebox{\linewidth}{!}{
		\begin{tikzpicture}
		\begin{axis}[%
		ybar,
		bar width=30pt,
		grid=both,
		width=6.0in,
		height=2.7in,
		x tick label style={
		%rotate=45,
		 /pgf/number format/.cd,
		set thousands separator={},
		nodes near coords,
        point meta=explicit symbolic,
		 },
		at={(0in,0in)},
		%xlabel={},
		ylabel={Cost},
		%scale only axis,
		separate axis lines,
		%every outer x axis line/.append style={white!15!black},
		%every x tick label/.append style={font=\color{white!15!black}},
		ymin=0,
		ymax="""+str(max1)+r""",
		xtick=data,
		legend style={draw=white!15!black,leThresholdsgend cell align=left,legend pos=north est},
		%symbolic x coords={$\rho=20\%$,$\rho=47\%$,$\rho=67\%$,$\rho=74\%$,$\rho=87\%$},
		symbolic x coords={Thresholds,MDP,Truncated-MDP,Greedy,Greedy-window},
		]
		\addplot [fill=blue] coordinates {(Thresholds,"""+str(c5)+r""")["""+str(c5)+r"""] (MDP,"""+str(c4)+r""")["""+str(c4)+r"""] (Truncated-MDP,"""+str(c3)+r""")["""+str(c3)+r"""] (Greedy,"""+str(c1)+r""")["""+str(c1)+r"""] (Greedy-window,"""+str(c2)+r""")["""+str(c2)+r"""]};
		% \addplot coordinates {($\rho=87\%$,128) ($\rho=74\%$,111) ($\rho=67\%$,101) ($\rho=47\%$,70) ($\rho=20\%$,30) };
		% \addplot coordinates {($\rho=87\%$,127) ($\rho=74\%$,110) ($\rho=67\%$,100) ($\rho=47\%$,70) ($\rho=20\%$,30) };
		% \legend{Low variance arrivals,Medium variance arrivals,High variance arrivals}
		\end{axis}
		\end{tikzpicture}%
		}
		\caption[Cost comparison]{Cost comparison.}
		\label{ALL5:"""+str(time.time())+r"""}
		\end{figure}
		%============================================================
		"""
		f.write(page2+"\n")	

		max1=max(t1,t2,t3,t4,t5)*1.2
		page3=r"""
		%============================================================
		\begin{figure}[H]
		\centering
		\resizebox{\linewidth}{!}{
		\begin{tikzpicture}
		\begin{axis}[%
		ybar,
		bar width=30pt,
		grid=both,
		width=6.0in,
		height=2.7in,
		x tick label style={
		%rotate=45,
		 /pgf/number format/.cd,
		set thousands separator={},
		nodes near coords,
        point meta=explicit symbolic,
		 },
		at={(0in,0in)},
		%xlabel={},
		ylabel={Time in second},
		%scale only axis,
		separate axis lines,
		%every outer x axis line/.append style={white!15!black},
		%every x tick label/.append style={font=\color{white!15!black}},
		ymin=0,
		ymax="""+str(max1)+r""",
		xtick=data,
		legend style={draw=white!15!black,leThresholdsgend cell align=left,legend pos=north est},
		%symbolic x coords={$\rho=20\%$,$\rho=47\%$,$\rho=67\%$,$\rho=74\%$,$\rho=87\%$},
		symbolic x coords={Thresholds,MDP,Truncated-MDP,Greedy,Greedy-window},
		]
		\addplot [fill=red] coordinates {(Thresholds,"""+str(t5)+r""")["""+str(t5)+r"""] (MDP,"""+str(t4)+r""")["""+str(t4)+r"""] (Truncated-MDP,"""+str(t3)+r""")["""+str(t3)+r"""] (Greedy,"""+str(t1)+r""")["""+str(t1)+r"""] (Greedy-window,"""+str(t2)+r""")["""+str(t2)+r"""]};
		% \addplot coordinates {($\rho=87\%$,128) ($\rho=74\%$,111) ($\rho=67\%$,101) ($\rho=47\%$,70) ($\rho=20\%$,30) };
		% \addplot coordinates {($\rho=87\%$,127) ($\rho=74\%$,110) ($\rho=67\%$,100) ($\rho=47\%$,70) ($\rho=20\%$,30) };
		% \legend{Low variance arrivals,Medium variance arrivals,High variance arrivals}
		\end{axis}
		\end{tikzpicture}%
		}
		\caption[Computation-time comparison]{Computation-time comparison.}
		\label{ALL5:"""+str(time.time())+r"""}
		\end{figure}
		%============================================================
		"""
		f.write(page3+"\n")	

		max1=max(m1,m2,m3,m4,m5)*1.2
		page4=r"""
		%============================================================
		\begin{figure}[H]
		\centering
		\resizebox{\linewidth}{!}{
		\begin{tikzpicture}
		\begin{axis}[%
		ybar,
		bar width=30pt,
		grid=both,
		width=6.0in,
		height=2.7in,
		x tick label style={
		%rotate=45,
		 /pgf/number format/.cd,
		set thousands separator={},
		nodes near coords,
        point meta=explicit symbolic,
		 },
		at={(0in,0in)},
		%xlabel={},
		ylabel={Memory in MB},
		%scale only axis,
		separate axis lines,
		%every outer x axis line/.append style={white!15!black},
		%every x tick label/.append style={font=\color{white!15!black}},
		ymin=0,
		ymax="""+str(max1)+r""",
		xtick=data,
		legend style={draw=white!15!black,leThresholdsgend cell align=left,legend pos=north est},
		%symbolic x coords={$\rho=20\%$,$\rho=47\%$,$\rho=67\%$,$\rho=74\%$,$\rho=87\%$},
		symbolic x coords={Thresholds,MDP,Truncated-MDP,Greedy,Greedy-window},
		]
		\addplot [fill=orange] coordinates {(Thresholds,"""+str(m5)+r""")["""+str(m5)+r"""] (MDP,"""+str(m4)+r""")["""+str(m4)+r"""] (Truncated-MDP,"""+str(m3)+r""")["""+str(m3)+r"""] (Greedy,"""+str(m1)+r""")["""+str(m1)+r"""] (Greedy-window,"""+str(m2)+r""")["""+str(m2)+r"""]};
		% \addplot coordinates {($\rho=87\%$,128) ($\rho=74\%$,111) ($\rho=67\%$,101) ($\rho=47\%$,70) ($\rho=20\%$,30) };
		% \addplot coordinates {($\rho=87\%$,127) ($\rho=74\%$,110) ($\rho=67\%$,100) ($\rho=47\%$,70) ($\rho=20\%$,30) };
		% \legend{Low variance arrivals,Medium variance arrivals,High variance arrivals}
		\end{axis}
		\end{tikzpicture}%
		}
		\caption[Memory comparison]{Memory comparison.}
		\label{ALL5:"""+str(time.time())+r"""}
		\end{figure}
		%============================================================
		"""
		f.write(page4+"\n")	
		f.write(r"\end{document}"+"\n")	
################################################################################
def memory():
	# print process.memory_info()
	# sys.exit()
	return sum(process.memory_info())/float(2**20)
	return process.memory_info()[0]/float(2**20)
################################################################################
def detect_delimiter(csv_file):
	with open(csv_file, 'r') as f:
		header=f.readline()
		sep=[';',',','\t']
		count_sep=[(header.count(s),s)for s in sep]
		count_sep.sort()
		return count_sep[-1][1]

def get_closest_index(h,str):
	return h.index(max([[e,difflib.SequenceMatcher(None, str.lower(),\
		e.lower()).ratio()] for e in h],key=lambda x:x[1])[0])
################################################################################
def import_histogram_dict(csv_path):
	with open(csv_path, 'r') as f:
		reader = csv.reader(f,delimiter=detect_delimiter(csv_path))
		lst = map(lambda x: [int(x[0]),float(x[1])],reader)
	print1(('histogram size',len(lst)))
	dicte=dict(lst)
	s=sum(dicte.values())
	for k in dicte:
		dicte[k]/=s
	# print dicte
	return dicte
################################################################################
# def convolution(X,Y):
# 	Z={}
# 	for x in X:
# 		for y in Y:
# 			p=X[x]*Y[y]
# 			z=x+y
# 			try:Z[z]+=p
# 			except:Z[z]=p
# 	return Z
################################################################################
def convolution(X,Y):
	q=int(max(max(X.keys()),max(Y.keys())))
	Xl=[]
	Yl=[]
	for i in range(q+1):
		try:Xl+=[X[i]]
		except:Xl+=[0]
	for i in range(q+1):
		try:Yl+=[Y[i]]
		except:Yl+=[0]
	Zl=signal.convolve(Xl,Yl,mode='full')
	Z={}
	for i in range(len(Zl)):
		if Zl[i]>0:Z[i]=Zl[i]
	return Z
################################################################################
def sub(X,v):
	Z={}
	for x in X:
		p=X[x]
		z=max(0,x-v)
		try:Z[z]+=p
		except:Z[z]=p
	return Z
################################################################################
def mine(X,v):
	Z={}
	for x in X:
		p=X[x]
		z=min(x,v)
		try:Z[z]+=p
		except:Z[z]=p
	return Z
################################################################################
def expectation(X):
	e=0
	for x in X:e+=x*X[x]
	return e
################################################################################
def sum_proba(X):
	return sum(X.values())
################################################################################
def rounde(X):
	Z={}
	for x in X:
		Z[x]=round(X[x],2)
	return Z
################################################################################
def Greedy_window(real_arrival,A,b,M,d,cN,cM,cOn,h,w):
	m0=memory()
	t0=time.time()
	def kernel(T):
		Cost={}
		Action={}
		n0=T
		for m in range(M+1):
			N={n0:1.}
			acumulated_cost=0
			for k in range(w):
				Z=convolution(N,A)
				N=mine(sub(Z,d*m),b)
				acumulated_cost+=m*cM+cN*expectation(N)
			Cost[m]=acumulated_cost
		for m0 in range(M+1):
			cost_min=10**10
			m1=0
			for m in range(M+1):
				C=Cost[m]+cOn*max(0,(m-m0))
				if C<cost_min:
					cost_min=C
					m1=m
			Action[n0,m0]=m1
		return Action

	liste=[n for n in range(b+1)]
	cpt_cpu=max(0,multiprocessing.cpu_count()-1)
	res=Pool(cpt_cpu).map(kernel,liste)
	Action={}
	for Action1 in res:
		for n0,m0 in Action1:
			Action[n0,m0]=Action1[n0,m0]

	n0=0
	m0=0
	total_cost=0
	for slot in range(h):
		a=real_arrival[slot]
		# print a,n0,m0
		n1=int(min(b,max(0,n0+a-d*m0)))
		m1=Action[n0,m0]
		total_cost+=cM*m0+cN*n0+cOn*max(0,(m1-m0))
		n0=n1
		m0=m1
	# print'-->',total_cost,total_cost
	m1=memory()
	t1=time.time()
	return total_cost,t1-t0,m1-m0
	# return total_cost
################################################################################
def thresholds(real_arrival,A,b,M,d,cN,cM,cOn,h,coupling):
	m0=memory()
	t0=time.time()
	def figure(UD,title,b,ax):
		matrix1=[]
		for U in range(b):
			row=[]
			for D in range(b):row+=[0]
			matrix1+=[row]
		for U,D in UD:
			matrix1[U][D]=UD[U,D]
		g = np.array(matrix1)
		Mg=max(max(x) for x in matrix1)
		mg=min(min(x) for x in matrix1)
		ax.set_title(title)
		ax.set_xlabel('D')
		ax.set_ylabel('U')
		cax=ax.imshow(g,cmap=cm.jet)
		fig.colorbar(cax, ticks=[int((Mg-mg)*x/6)for x in range(7)],ax=ax)
		# plt.savefig('plot'+str(int(time.time()*1000))+'.png')
		# plt.show()
		return ax
	def kernel(T):
		epsilon=10**(-5)
		def equals(X,Y):
			if len(X)!=len(Y):return False
			for i in X:
				if i not in Y:return False
				if abs(X[i]-Y[i])>epsilon:return False           
			return True
		U,D=T
		N={0:1.}
		# Nsup={b:1.}
		# Ninf={0:1.}
		total_cost=0
		m=0
		QoS=0
		Energy=0
		for slot in range(h):
			E=expectation(N)
			if E<=D:
				m=max(m-1,0)
				# Nsup={b:1.}
				# Ninf={0:1.}
			elif E>=U:
				m=min(m+1,M)
				switch_on_cost=cOn
				Energy+=switch_on_cost
				# Nsup={b:1.}
				# Ninf={0:1.}
			Energy+=m*cM
			QoS+=cN*expectation(N)
			Z=convolution(N,A)
			N=mine(sub(Z,d*m),b)

			# if coupling:
			# 	if equals(Nsup,Ninf):
			# 		n=expectation(N)
			# 		for slot1 in range(slot+1,h):
			# 			Energy+=m*cM
			# 			QoS+=cN*n
			# 		break
			# 	Nsup=mine(sub(convolution(Nsup,A),d*m),b)
			# 	Ninf=mine(sub(convolution(Ninf,A),d*m),b)
		total_cost=Energy+QoS
		return(U,D,total_cost,QoS,Energy)

	# liste=[(U,D)for U in range(0,b,int(b**.5)) for D in range(0,U+1,int(b**.5))]
	liste=[(U,D)for U in range(b) for D in range(U+1)]
	cpt_cpu=max(0,multiprocessing.cpu_count()-1)
	print cpt_cpu
	res=Pool(cpt_cpu).map(kernel,liste)
	UD={}
	UD_QoS={}
	UD_Energy={}
	UD_seuil={}
	for(U,D,total_cost,QoS,Energy)in res:
		UD[U,D]=total_cost
		UD_QoS[U,D]=QoS
		UD_Energy[U,D]=Energy

	U,D=min(UD.iteritems(),key=operator.itemgetter(1))[0]
	cost_min=min(UD.values())
	for k in UD:
		if D<=U and cost_min*.95<=UD[k]<=cost_min*1.05:
			UD_seuil[k]=1
		else:UD_seuil[k]=2
	print (U,D)
	fig, _axs = plt.subplots(nrows=2, ncols=2)
	# fig.subplots_adjust(hspace=0.3)
	axs = _axs.flatten()
	figure(UD_Energy,'Energy',b,axs[0])
	figure(UD_QoS,'QoS',b,axs[1])
	figure(UD,'Cost',b,axs[2])
	figure(UD_seuil,'Result',b,axs[3])
	fig.tight_layout()
	png1=TT+'plot'+'.png'
	plt.savefig(png1)
	# plt.show()

	n0=0
	m0=0
	total_cost=0
	for slot in range(h):
		# print n0,m0
		a=real_arrival[slot]
		n1=int(min(b,max(0,n0+a-d*m0)))
		m1=m0
		if n1>=U:
			m1=min(m0+1,M)
		elif n1<=D:
			m1=max(m0-1,0)
		total_cost+=cM*m0+cN*n0+cOn*max(0,(m1-m0))
		n0=n1
		m0=m1
	# print'-->',cost_min,U,D,total_cost
	m1=memory()
	t1=time.time()
	return total_cost,t1-t0,m1-m0,png1,plt
	# return cost_min,cost_min/h,U,D
################################################################################
def mdp_prism_code(real_arrival,A,b,M,d,cN,cM,cOn,h):
	file=TT+'mpd'+'.prism'
	print1(('PRISM code',file))
	with open(file,"w") as f:
		f.write("mdp\n")	
		f.write("const int B="+str(b)+";\n")	
		f.write("const int m="+str(M)+";\n")	
		f.write("const int d="+str(d)+";\n")	
		f.write("const int cN="+str(cN)+";\n")	
		f.write("const int cM="+str(cM)+";\n")	
		f.write("const int cOn="+str(cOn)+";\n")
		i=0
		for k in A:
			f.write("const int d"+str(i)+"="+str(int(k))+";\n")
			f.write("const double p"+str(i)+"="+str(A[k])+";\n")
			i+=1	
		f.write("module system1\n")	
		f.write("M : [0..m] init 0;\n")	
		f.write("N : [0..B] init 0;\n")	
		for m in range(M+1):
			f.write("[a"+str(m)+"]  true ->\n")	
			line=""
			i=0
			for k in A:
				K=str(int(i))
				line+="p"+K+":(N'=min(B,max(0,N+d"+K+"-M*d)))&(M'="+str(m)+")+\n"
				i+=1	
			line=line[:-2]+";\n"
			f.write(line)	
		f.write("endmodule\n")	
		f.write('rewards "r"\n')	
		for m in range(M+1):
			f.write("[a"+str(m)+"]  true : M*cM+N*cN+max(0,"+str(m)+"-M)*cOn;\n")	
		f.write("endrewards\n") 
	print'use file',file
	print'use prism cmd','Rmin=?[C<='+str(int(h))+']'
################################################################################
def mdp(real_arrival,A,b,M,d,cN,cM,cOn,h):
	m0=memory()
	t0=time.time()
	S=(M+1)*(b+1)
	cpt=0
	index1={}
	P = [None] * (M+1)
	R = [None] * (M+1)
	for act in range(M+1):
		m1=act
		dictProba={}
		dictCost={}
		for n0 in range(b+1):
			for m0 in range(M+1):
				try:
					i=index1[(n0,m0)]
				except:
					i=cpt
					index1[(n0,m0)]=cpt
					cpt+=1
				dictCost[i]=-(cM*m0+cN*n0+cOn*max(0,(m1-m0)))
				for arivee in A:
					n1=int(min(b,max(0,n0+arivee-d*m0)))

					try:
						j=index1[(n1,m1)]
					except:
						j=cpt
						index1[(n1,m1)]=cpt
						cpt+=1

					proba=A[arivee]
					try:
						dictProba[(i,j)]+=proba
					except:
						dictProba[(i,j)]=proba
				# print (index1[(n0,m0)],m0,n0),'-->',act,'-->',(index1[(n1,m1)],m1,n1)
		data=[]
		I=[]
		J=[]
		for i,j in dictProba:
			data.append(dictProba[(i,j)])
			I.append(i)
			J.append(j)
		ij=[I,J]
		# P[act]=sparse((data, ij),shape=(S, S)).todense()
		P[act]=sparse((data, ij),shape=(S, S))
		
		data=[]
		for i in dictCost:
			data.append(dictCost[i])
		R[act]=data
	R = np.array(R).transpose()
	# print ('P',P)
	# print ('R',R)
	vi = mdptoolbox.mdp.FiniteHorizon(P,R,1,h)
	vi.run()
	print vi.V.transpose()[::-1]
	# print vi.policy
	# print vi.policy[index1[(0,0)]]
	cost_min=-vi.V[index1[(0,0)]][0]
	# print vi.V[index1[(0,0)]]
	# print vi.V[index1[(0,0)]][0]
	n0=0
	m0=0
	total_cost=0
	for slot in range(h):
		# print n0,m0
		a=real_arrival[slot]
		n1=int(min(b,max(0,n0+a-d*m0)))
		# m1=vi.policy[index1[(n0,m0)]][-slot-1]
		m1=vi.policy[index1[(n0,m0)]][slot]
		total_cost+=cM*m0+cN*n0+cOn*max(0,(m1-m0))
		n0=n1
		m0=m1
	print'mdp: real cost -->',total_cost
	print'mdp: expected cost -->',cost_min
	# return total_cost
	# return cost_min
	m1=memory()
	t1=time.time()
	return cost_min,t1-t0,m1-m0
################################################################################
def mdp_truncated(real_arrival,A,b,M,d,cN,cM,cOn,h,w):
	m0=memory()
	t0=time.time()
	S=(M+1)*(b+1)
	cpt=0
	index1={}
	P = [None] * (M+1)
	R = [None] * (M+1)
	for act in range(M+1):
		m1=act
		dictProba={}
		dictCost={}
		for n0 in range(b+1):
			for m0 in range(M+1):
				try:
					i=index1[(n0,m0)]
				except:
					i=cpt
					index1[(n0,m0)]=cpt
					cpt+=1
				dictCost[i]=-(cM*m0+cN*n0+cOn*max(0,(m1-m0)))
				for arivee in A:
					n1=int(min(b,max(0,n0+arivee-d*m0)))

					try:
						j=index1[(n1,m1)]
					except:
						j=cpt
						index1[(n1,m1)]=cpt
						cpt+=1

					proba=A[arivee]
					try:
						dictProba[(i,j)]+=proba
					except:
						dictProba[(i,j)]=proba
				# print (index1[(n0,m0)],m0,n0),'-->',act,'-->',(index1[(n1,m1)],m1,n1)
		data=[]
		I=[]
		J=[]
		for i,j in dictProba:
			data.append(dictProba[(i,j)])
			I.append(i)
			J.append(j)
		ij=[I,J]
		# P[act]=sparse((data, ij),shape=(S, S)).todense()
		P[act]=sparse((data, ij),shape=(S, S))
		
		data=[]
		for i in dictCost:
			data.append(dictCost[i])
		R[act]=data
	R = np.array(R).transpose()
	# print ('P',P)
	# print ('R',R)
	vi = mdptoolbox.mdp.FiniteHorizon(P,R,1,w)
	vi.run()
	# print vi.V
	# print vi.policy
	# print vi.policy[index1[(0,0)]]
	cost_min=-vi.V[index1[(0,0)]][0]
	# print vi.V[index1[(0,0)]]
	# print vi.V[index1[(0,0)]][0]
	n0=0
	m0=0
	total_cost=0
	for slot in range(h):
		# print n0,m0
		a=real_arrival[slot]
		n1=int(min(b,max(0,n0+a-d*m0)))
		# m1=vi.policy[index1[(n0,m0)]][-slot-1]
		m1=vi.policy[index1[(n0,m0)]][0]
		total_cost+=cM*m0+cN*n0+cOn*max(0,(m1-m0))
		n0=n1
		m0=m1
	# print'-->',cost_min,total_cost
	m1=memory()
	t1=time.time()
	return total_cost,t1-t0,m1-m0
	# return total_cost
################################################################################
# procédure qui intercepte l'action de clique sur un bouton de 
# l'interface graphique
################################################################################
def press(btn):
	global Directory
	global A
	global b
	global M
	global d
	global cN
	global cM
	global cOn
	global h
	global w
	global real_arrival
	global histogram
	global c1,t1,m1
	global c2,t2,m2
	global c3,t3,m3
	global c4,t4,m4
	global c5,t5,m5,png1
	global m0

	# simulation0
	real_arrival=[]
	histogram=A
	if btn=="Quitter":
		app.stop()
	elif btn=="Histogram des arrivées":
		Directory=app.openBox(title=None, dirName=None, fileTypes=[('CSV', '*.csv'), ('Text', '*.txt')], asFile=False, parent=None, multiple=False, mode='r')
		s=Directory
		s=os.path.basename(Directory)
		app.setLabel("A",s)
	elif btn=="Calculer":
		print1("--------------------")
		A=import_histogram_dict(Directory)
		print1(('Expectation',expectation(A)))
		b=int(app.getEntry("Buffer size............"))
		M=int(app.getEntry("Max number of servers.."))
		d=int(app.getEntry("Service rate..........."))
		h=int(app.getEntry("Horizon................"))
		w=int(app.getEntry("Window................."))
		cN=float(app.getEntry("Waiting cost..........."))
		cM=float(app.getEntry("Serving cost..........."))
		cOn=float(app.getEntry("Switching on cost......"))
		methods=app.getProperties("Methods")
		option2=app.getProperties("More options")

		# simulation0
		real_arrival=[]
		histogram=A
		random.seed(0)
		np.random.seed(0)
		for slot in range(h):
			a=np.random.choice(histogram.keys(),p=histogram.values())
			real_arrival+=[a]

		c1,t1,m1=1,1,1
		c2,t2,m2=1,1,1
		c3,t3,m3=1,1,1
		c4,t4,m4=1,1,1
		c5,t5,m5,png1=1,1,1,"none.png"

		if option2["Output PRISM code"]:
			mdp_prism_code(real_arrival,A,b,M,d,cN,cM,cOn,h)
		
		m0=memory()

		if methods["Greedy"]:
			c1,t1,m1=Greedy_window(real_arrival,A,b,M,d,cN,cM,cOn,h,1)
			print1(('#Greedy       ',c1,t1,m1))

		if methods["Greedy window"]:
			c2,t2,m2=Greedy_window(real_arrival,A,b,M,d,cN,cM,cOn,h,w)
			print1(('#Greedy_window',c2,t2,m2))

		if methods["Truncated MDP"]:
			try:
				c3,t3,m3=mdp_truncated(real_arrival,A,b,M,d,cN,cM,cOn,h,w)
				print1(('#mdp_truncated',c3,t3,m3))
			except:
				c3,t3,m3=1,1,1
				print1(('#mdp_truncated',c3,t3,m3))

		if methods["MDP"]:
			try:
				c4,t4,m4=mdp(real_arrival,A,b,M,d,cN,cM,cOn,h)
				print1(('#mdp          ',c4,t4,m4))
			except:
				c4,t4,m4=1,1,1
				print1(('#mdp          ',c4,t4,m4))

		if methods["Thresholds"]:
			try:
				c5,t5,m5,png1,plt=thresholds(real_arrival,A,b,M,d,cN,cM,cOn,h,coupling=option2["Coupling"])
				print1(('#thresholds   ',c5,t5,m5))
				plt.show()
				# app.setImageSize("image", 640, 480)
				# app.reloadImage("image",png1)
				# img=mpimg.imread(png1)
				# imgplot = plt.imshow(img)
				# plt.show()				
			except:
				c5,t5,m5,png1=1,1,1,"none.png"
				print1(('#thresholds   ',c5,t5,m5))

		if option2["Output results as LaTex"]:
			latex_arrivee(real_arrival,A,b,M,d,cN,cM,cOn,h,w,c1,c2,c3,c4,c5,t1,t2,t3,t4,t5,m1,m2,m3,m4,m5,png1)
		print1("--------------------")
		# # directory=app.getLabel('ldossier')
		# year   =app.getOptionBox("Année     ")
		# month  =app.getOptionBox("Mois      ")
		# version=app.getOptionBox("Version   ")
		# recalage2(Directory,year,month,version)
		# app.addListItems('log',[ts_to_local_time_str(time())+' '+"fin"])
		# app.infoBox("Recalage", "Procédure de recalage terminée", parent=None)
		None
	else:
		None
################################################################################
# composition de l'interface graphique
################################################################################
process = psutil.Process(os.getpid())
app=gui()
app.setLocation(0,0)
app.setResizable(canResize=False)
# app.setFont(size=11, family="Courier", underline=False, slant="italic")
app.setFont(size=11,family="Courier")
app.setSticky("news")
app.setExpand("both")

app.addLabelEntry("Buffer size............", colspan=2)
app.addLabelEntry("Max number of servers..", colspan=2)
app.addLabelEntry("Service rate...........", colspan=2)
app.addLabelEntry("Waiting cost...........", colspan=2)
app.addLabelEntry("Serving cost...........", colspan=2)
app.addLabelEntry("Switching on cost......", colspan=2)
app.addLabelEntry("Horizon................", colspan=2)
app.addLabelEntry("Window.................", colspan=2)
app.addButton("Histogram des arrivées",press,colspan=2)
app.addLabel("A", "", colspan=2)
row = app.getRow() # get current row
methods={
"MDP":False, 
"Truncated MDP":False, 
"Thresholds":False, 
"Greedy":False,
"Greedy window":False,
}
option2={
"Coupling":True, 
"Output PRISM code":True, 
"Output results as LaTex":True, 
}
app.addProperties("Methods", methods,  row, 0)
app.addProperties("More options", option2,  row, 1)

app.addListBox("log",["Results"], colspan=2) 
app.addButton("Calculer",press, colspan=2)
app.addButton("Quitter",press, colspan=2)
# app.addImage("image", "none.png",0,2, rowspan=11)
################################################################################
# lancement de l'interface graphique
################################################################################
app.go()
################################################################################
