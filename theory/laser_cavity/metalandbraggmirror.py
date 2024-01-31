# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Metalmirror.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!
#! /usr/bin/env python2.7
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import pyqtRemoveInputHook

from numpy import *
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import time
import os
from math import pi as Pi
from bisect import insort_left

#----------Programm---------------
#================Refractive index============
AU_N=load('AU_BABAR_N.npy')
AU_K=load('AU_BABAR_K.npy')
AU_L=load('AU_BABAR_L.npy')
#print(AU_N)
AG_N=load('AG_BABAR_N.npy')
AG_K=load('AG_BABAR_K.npy')
AG_L=load('AG_BABAR_L.npy')

#================Functions=====================
def ref_index(nlambda,matN,matK,matLambda):
	nlambda=nlambda/1000 #data in um
	I_sec=argsort(abs(matLambda-nlambda))[1]
	I_first = argsort(abs(matLambda-nlambda))[0]
	L_sec=matLambda[I_sec]
	L_first=matLambda[I_first]
	
	#n.real
	sec=matN[I_sec]
	first=matN[I_first]   
	n=poly1d(polyfit([L_first,L_sec],[first,sec],1))(nlambda)
	#n.imag
	sec=matK[I_sec]
	first=matK[I_first]
	k=poly1d(polyfit([L_first,L_sec],[first,sec],1))(nlambda)
	#print(nlambda,n+1j*k)
	return n+1j*k
def Mij(from_n,to_n):
    return array([[from_n+to_n,-from_n+to_n],[-from_n+to_n,from_n+to_n]])/(2*to_n)

def Mi(n,d,lambda0,g=0):
    k0=2*pi/lambda0
    phi=n*k0*d
    return array([[exp(1j*phi+g*d),0],[0,exp(-1j*phi-g*d)]])

def S(M):
    deter = linalg.det(M)
    if abs(M[1,1])<10e-7:
        print ("unstable")
    return array([[deter,M[0,1]],[-M[1,0],1]]) / M[1,1]
def Tra(S,n_start,n_end):
    #Only for: alpha out=0=alpha in
    return (n_end/n_start).real * absolute(S[0,0])**2
def Ref(S):
    return absolute(S[1,0])**2
def RefShift(S):
    return angle(S[1,0],deg=0)

#================Parameter==================
gain_finesse=1000000
pyqtRemoveInputHook()
gold = sqrt(-2.8159+1.j*3.3686)#53nm thick, 500nm
Glass = sqrt(3.5) #SF11 500nm
hBn = 2.07 #500nm
yag = 1.84 #500nm
#--n1--n2--n3--n4--n5--n6--n7
#--d1--d2--d3--d4--d5--d6--d7


#================Calculation==================
def switch(ns,ds,g,lambdas,BraggNs,Win):
	
	if BraggNs[0]==-1 and BraggNs[1]==-1: #Both are Metal Mirrors
		return run_Metal_Metal(ns,ds,g,lambdas,Win)
	elif BraggNs[0]!=-1 and BraggNs[1]!=-1: #Both are Bragg Mirrors
		return run_Bragg_Bragg(ns,ds,g,lambdas,BraggNs,Win)
	elif BraggNs[0]!=-1 and BraggNs[1]==-1: #Left Mirror is Bragg, Right Mirror is Metal
		return run_Bragg_Metal(ns,ds,g,lambdas,BraggNs,Win)
	elif BraggNs[0]==-1 and BraggNs[1]!=-1: #Left Mirror is Metal, Right Mirror is Bragg
		return run_Metal_Bragg(ns,ds,g,lambdas,BraggNs,Win)
	else:
		print 'TO DO'
#switch(self.n_array,self.d_array,self.g,self.lambdas,self.NBR_array,self)    

def run_Bragg_Bragg(ns,ds,g,lambdas,BraggNs,Win):
	#refractive index
	n1	=	complex(ns[0])
	n2a	=	complex(ns[1])
	n2b	=	complex(ns[2])
	n3	=	complex(ns[3])
	n4	=	complex(ns[4])
	n5	=	complex(ns[5])
	n6a	=	complex(ns[6])
	n6b	=	complex(ns[7])
	n7	=	complex(ns[8])
	
	#Gain
	g=float(g)/gain_finesse
	
	#Distances
	d2	=	float(ds[0]) #ignored
	d3	=	float(ds[1])
	d4	=	float(ds[2])
	d5	=	float(ds[3])
	d6	=	float(ds[4]) #ignored
	
	#Overwrite and create Distances
	d2a =	Win.lambda0/(4*n2a.real)
	d2b	=	Win.lambda0/(4*n2b.real)
	d6a =	Win.lambda0/(4*n6a.real)
	d6b	=	Win.lambda0/(4*n6b.real)
	
	#Numbers of Bragg Elements
	n_L	=	BraggNs[0]
	n_R	=	BraggNs[1]
	#Output
	cls()
	print 'Simulation Bragg Mirror - Bragg Mirror' 
	print ''	
	print 'n0 = ',n1
	print'n1a= ',n2a
	print'n1b= ',n2b
	print'n2 = ',n3
	print'n3 = ',n4
	print'n4 = ',n5
	print'n5a= ',n6a
	print'n5b= ',n6b
	print'n6 = ',n7
	print ''
	print 'd1a= ',d2a
	print 'd1b= ',d2b
	print 'd2= ',d3
	print 'd3= ',d4
	print 'd4= ',d5
	print 'd5a= ',d6a
	print 'd5b= ',d6b
	print 'N_L= ',n_L
	print 'N_R=', n_R
	print ''
	print 'g= ',g
	#INI return arrays
	Roundtrip=array([])
	Roundtripphase=array([])
	Roundtrip_out_left = array([])
	Roundtrip_out_right = array([])
	Q_fac = array([])
	
	#Boundary Matrices
	M54=Mij(n5,n4)
	M43=Mij(n4,n3)
	M32a=Mij(n3,n2a)
	M2a2b=Mij(n2a,n2b)
	M2b2a=Mij(n2b,n2a)
	M56a=Mij(n5,n6a)
	M6a6b=Mij(n6a,n6b)
	M6b6a=Mij(n6b,n6a)
	
	if Win.auto_gain:
		lambdas=array([Win.lambda0*1.0])
	for lam in lambdas:
		#Free Propagation Matrices		
		M4=Mi(n4,d4,lam,g)		
		M3=Mi(n3,d3,lam,0)		
		M2a=Mi(n2a,d2a,lam,0)		
		M2b=Mi(n2b,d2b,lam,0)		
		#Single Bragg Element
		Mb_L=dot(M2b2a,dot(M2b,dot(M2a2b,M2a)))
		#Multiple Bragg Elements
		Bragg_L=linalg.matrix_power(Mb_L,n_L-1)
		#Substrate
		M21=Mij(n2b,n1)
		Mbragg1=dot(M21,dot(M2b,dot(M2a2b,dot(M2a,Bragg_L))))
		try:		
			Win.progressBar.setValue(min(100/(lambdas[-1]-lambdas[0])*(lam-lambdas[0]),100))
		except:
			print()

		#Free Propagation Matrices
		M5=Mi(n5,d5,lam,0)		
		M6a=Mi(n6a,d6a,lam,0)		
		M6b=Mi(n6b,d6b,lam,0)		
		#Single Bragg Element
		Mb_R=dot(M6b6a,dot(M6b,dot(M6a6b,M6a)))
		#Multiple Bragg Elements
		Bragg_R=linalg.matrix_power(Mb_R,n_R-1)
		#Substrate
		M67=Mij(n6b,n7)
		Mbragg2=dot(M67,dot(M6b,dot(M6a6b,dot(M6a,Bragg_R))))

		Mtot_right=dot(Mbragg2,dot(M56a,M5))
		Stot_right=S(Mtot_right)

		Mtot_left=dot(Mbragg1,dot(M32a,dot(M3,dot(M43,dot(M4,M54)))))
		Stot_left=S(Mtot_left)
			
		v_start = array([[1],[0]])
		v_left = dot(Stot_left,v_start)

		v_start_rigth = array([[v_left[1,0]],[0]])
		v_right =dot(Stot_right,v_start_rigth) 

		#---------Auto Section--------------
		
		if int(lam)==int(Win.lambda0):
			angl = angle(v_right[1,0],deg=1)
			ampl = absolute(v_right[1,0])**2
			
			if Win.auto_gain:
				if abs(ampl-1)<=1e-3:
					Win.auto_gain=False
					return [0,0]
				else:
					return [-1,ampl-1]
			elif Win.auto_phase:
				if absolute(angl)<1e-4:
					Win.auto_phase=False
					return [0,0]
				else:
					return [-2,angl]#/((phase_right-phase_left)/2.0)]
		#---------------------------------		

		Roundtrip=append(Roundtrip,absolute(v_right[1,0])**2)
		Roundtripphase = append(Roundtripphase,angle(v_right[1,0],deg=1))
		Roundtrip_out_left =append(Roundtrip_out_left, (n1/n3).real * absolute(v_left[0,0])**2)
		Roundtrip_out_right=append(Roundtrip_out_right,(n7/n3).real * absolute( v_right[0,0])**2)
		Q_fac=append(Q_fac,-4*Pi/(lam)*(n3*d3+n5*d5+n4*d4)/log(Roundtrip[-1]))	

	return [Roundtrip,Roundtripphase,Roundtrip_out_left,Roundtrip_out_right,Q_fac]
	
def run_Metal_Bragg(ns,ds,g,lambdas,BraggNs,Win):
	#refractive index
	n1	=	complex(ns[0])
	#n2	=	complex(ns[1])
	#n2b	=	complex(ns[2])
	n3	=	complex(ns[3])
	n4	=	complex(ns[4])
	n5	=	complex(ns[5])
	n6a	=	complex(ns[6])
	n6b	=	complex(ns[7])
	n7	=	complex(ns[8])
	
	#Gain
	g=float(g)/gain_finesse
	
	#Distances
	d2	=	float(ds[0])
	d3	=	float(ds[1])
	d4	=	float(ds[2])
	d5	=	float(ds[3])
	d6	=	float(ds[4]) #ignored
	
	#Overwrite and create Distances
	d6a =	Win.lambda0/(4*n6a.real)
	d6b	=	Win.lambda0/(4*n6b.real)
	
	#Numbers of Bragg Elements
	n_R	=	BraggNs[1]
	#Output
	cls()
	print 'Simulation Metal Mirror - Bragg Mirror' 
	print ''	
	print 'n0 = ',n1
	print'n1= ',ns[1]
	print'n2 = ',n3
	print'n3 = ',n4
	print'n4 = ',n5
	print'n5a= ',n6a
	print'n5b= ',n6b
	print'n6 = ',n7
	print ''
	print 'd1= ',d2
	print 'd2= ',d3
	print 'd3= ',d4
	print 'd4= ',d5
	print 'd5a= ',d6a
	print 'd5b= ',d6b
	print 'N_R=', n_R
	print ''
	print 'g= ',g
	#INI return arrays
	Roundtrip=array([])
	Roundtripphase=array([])
	Roundtrip_out_left = array([])
	Roundtrip_out_right = array([])
	Q_fac = array([])
	
	if Win.auto_gain or  Win.auto_phase:
		lambdas=array([Win.lambda0*1.0])
	for lam in lambdas:
		if ns[1]=='AU':
			n2=ref_index(lam,AU_N,AU_K,AU_L)
			#if lam==300 or lam==350 or lam==400 or lam==450 or  lam==500  or lam==600 or lam==700:
				#print(n2)
		elif ns[1]=='AG':
			n2=ref_index(lam,AG_N,AG_K,AG_L)
			#if lam==300 or lam==350 or lam==400 or lam==450 or  lam==500  or lam==600 or lam==700:
				#print(n2)
		else:
			n2=complex(ns[1])
			#Boundary Matrices
			
		try:		
			Win.progressBar.setValue(min(100/(lambdas[-1]-lambdas[0])*(lam-lambdas[0]),100))
		except:
			print()
		M54=Mij(n5,n4)
		M43=Mij(n4,n3)
		M32=Mij(n3,n2)
		M56a=Mij(n5,n6a)
		M6a6b=Mij(n6a,n6b)
		M6b6a=Mij(n6b,n6a)
		#Free Propagation Matrices		
		M4=Mi(n4,d4,lam,g)		
		M3=Mi(n3,d3,lam,0)		
		M2=Mi(n2,d2,lam,0)
		M21=Mij(n2,n1)


		#Free Propagation Matrices
		M5=Mi(n5,d5,lam,0)		
		M6a=Mi(n6a,d6a,lam,0)		
		M6b=Mi(n6b,d6b,lam,0)		
		#Single Bragg Element
		Mb_R=dot(M6b6a,dot(M6b,dot(M6a6b,M6a)))
		#Multiple Bragg Elements
		Bragg_R=linalg.matrix_power(Mb_R,n_R-1)
		#Substrate
		M67=Mij(n6b,n7)
		Mbragg2=dot(M67,dot(M6b,dot(M6a6b,dot(M6a,Bragg_R))))



		Mtot_right=dot(Mbragg2,dot(M56a,M5))
		Stot_right=S(Mtot_right)

		Mtot_left=dot(M21,dot(M2,dot(M32,dot(M3,dot(M43,dot(M4,M54))))))
		Stot_left=S(Mtot_left)

		v_start = array([[1],[0]])
		v_left = dot(Stot_left,v_start)

		v_start_rigth = array([[v_left[1,0]],[0]])
		v_right =dot(Stot_right,v_start_rigth) 
		
				#---------Auto Section--------------
		
		if int(lam)==int(Win.lambda0):
			angl = angle(v_right[1,0],deg=1)
			ampl = absolute(v_right[1,0])**2
			if Win.auto_gain:
				print ampl
				if abs(ampl-1)<=1e-2:
					Win.auto_gain=False
					return [0,0]
				else:
					return [-1,ampl-1]
			elif Win.auto_phase:
				if absolute(angl)<0.1:
					Win.auto_phase=False
					return [0,0]
				else:
					return [-2,angl]#/((phase_right-phase_left)/2.0)]
		#---------------------------------		
					
		
		Roundtrip=append(Roundtrip,absolute(v_right[1,0])**2)
		Roundtripphase = append(Roundtripphase,angle(v_right[1,0],deg=1))
		Roundtrip_out_left =append(Roundtrip_out_left, (n1/n3).real * absolute(v_left[0,0])**2)
		Roundtrip_out_right=append(Roundtrip_out_right,(n7/n3).real * absolute( v_right[0,0])**2)
		Q_fac=append(Q_fac,-4*Pi/(lam)*(n3*d3+n5*d5+n4*d4)/log(Roundtrip[-1]))	

	return [Roundtrip,Roundtripphase,Roundtrip_out_left,Roundtrip_out_right,Q_fac]
		
def run_Bragg_Metal(ns,ds,g,lambdas,BraggNs,Win):
	#refractive index
	n1	=	complex(ns[0])
	n2a	=	complex(ns[1])
	n2b	=	complex(ns[2])
	n3	=	complex(ns[3])
	n4	=	complex(ns[4])
	n5	=	complex(ns[5])
	#n6	=	complex(ns[6])
	#n6b	=	complex(ns[7])
	n7	=	complex(ns[8])
	
	#Gain
	g=float(g)/gain_finesse
	print('fn',g)
	#Distances
	d2	=	float(ds[0]) #ignored
	d3	=	float(ds[1])
	d4	=	float(ds[2])
	d5	=	float(ds[3])
	d6	=	float(ds[4]) 
	
	#Overwrite and create Distances
	d2a =	Win.lambda0/(4*n2a.real)
	d2b	=	Win.lambda0/(4*n2b.real)
	
	#Numbers of Bragg Elements
	n_L	=	BraggNs[0]
	#Output
	#cls()
	print 'Simulation Bragg Mirror - Metal Mirror' 
	print ''	
	print 'n0 = ',n1
	print'n1a= ',n2a
	print'n1b= ',n2b
	print'n2 = ',n3
	print'n3 = ',n4
	print'n4 = ',n5
	print'n5= ',ns[6]
	print'n6 = ',n7
	print ''
	print 'd1a= ',d2a
	print 'd1b= ',d2b
	print 'd2= ',d3
	print 'd3= ',d4
	print 'd4= ',d5
	print 'd5= ',d6
	print 'N_L= ',n_L
	print ''
	print 'g= ',g
	#INI return arrays
	Roundtrip=array([])
	Roundtripphase=array([])
	Roundtrip_out_left = array([])
	Roundtrip_out_right = array([])
	Q_fac = array([])
	if Win.auto_gain or  Win.auto_phase:
		lambdas=array([Win.lambda0*1.0])
	
	for lam in lambdas:
		if ns[6]=='AU':
			n6 = ref_index(lam,AU_N,AU_K,AU_L)	
		elif ns[6]=='AG':
			n6 = ref_index(lam,AG_N,AG_K,AG_L)
		else:
			n6=complex(ns[6])
		try:		
			Win.progressBar.setValue(min(100/(lambdas[-1]-lambdas[0])*(lam-lambdas[0]),100))
		except:
			print()
		#Boundary Matrices
		M54=Mij(n5,n4)
		M43=Mij(n4,n3)
		M32a=Mij(n3,n2a)
		M2a2b=Mij(n2a,n2b)
		M2b2a=Mij(n2b,n2a)
		M56=Mij(n5,n6)
		M67=Mij(n6,n7)
		#Free Propagation Matrices		
		M4=Mi(n4,d4,lam,g)		
		M3=Mi(n3,d3,lam,0)		
		M2a=Mi(n2a,d2a,lam,0)		
		M2b=Mi(n2b,d2b,lam,0)		
		#Single Bragg Element
		Mb_L=dot(M2b2a,dot(M2b,dot(M2a2b,M2a)))
		#Multiple Bragg Elements
		Bragg_L=linalg.matrix_power(Mb_L,n_L-1)
		#Substrate
		M21=Mij(n2b,n1)
		Mbragg1=dot(M21,dot(M2b,dot(M2a2b,dot(M2a,Bragg_L))))


		#Free Propagation Matrices
		M5=Mi(n5,d5,lam,0)		
		M6=Mi(n6,d6,lam,0)		
		
		Mtot_right=dot(M67,dot(M6,dot(M56,M5)))		
		Stot_right=S(Mtot_right)

		Mtot_left=dot(Mbragg1,dot(M32a,dot(M3,dot(M43,dot(M4,M54)))))
		Stot_left=S(Mtot_left)
		v_start = array([[1],[0]])
		v_left = dot(Stot_left,v_start)

		v_start_rigth = array([[v_left[1,0]],[0]])
		v_right =dot(Stot_right,v_start_rigth) 
		
		
		
				#---------Auto Section--------------
		
		if int(lam)==int(Win.lambda0):
			angl = angle(v_right[1,0],deg=1)
			#print absolute(v_right[1,0])**2
			#quest=raw_input('Do you want to continue? Y/N   ')	
			#ampl = absolute(v_right[1,0])**2
			if Win.auto_gain:
				if abs(absolute(v_right[1,0])**2-1)<=1e-3:
					Win.auto_gain=False
					return [0,0]
				else:
					return [-1,absolute(v_right[1,0])**2-1]
			elif Win.auto_phase:
				if absolute(angl)<1e-4:
					Win.auto_phase=False
					return [0,0]
				else:
					return [-2,angl]#/((phase_right-phase_left)/2.0)]
		#---------------------------------		

		Roundtrip=append(Roundtrip,absolute(v_right[1,0])**2)
		Roundtripphase = append(Roundtripphase,angle(v_right[1,0],deg=1))
		Roundtrip_out_left =append(Roundtrip_out_left, (n1/n3).real * absolute(v_left[0,0])**2)
		Roundtrip_out_right=append(Roundtrip_out_right,(n7/n3).real * absolute( v_right[0,0])**2)
		Q_fac=append(Q_fac,-4*Pi/(lam)*(n3*d3+n5*d5+n4*d4)/log(Roundtrip[-1]))	

	return [Roundtrip,Roundtripphase,Roundtrip_out_left,Roundtrip_out_right,Q_fac]

def run_Metal_Metal(ns,ds,g,lambdas,Win):
	#--n1--n2--n3--n4--n5--n6--n7
	#--d1--d2--d3--d4--d5--d6--d7
	#print_array_n=['n0= ','n1= ','n2= ','n3= ','n4= ','n5= ','n6= ']
	print_array_d=['d1= ','d2= ','d3= ','d4= ','d5= ']
	#time.sleep(5)
	#print(ds)
	#quest=raw_input('Do you want to continue? Y/N   ')	
	try:
		n1=complex(ns[0])
	except:
		return [0,0,0,0,0]
	

	#n2	=	complex(ns[1])
	#n2b	=	complex(ns[2])
	n3	=	complex(ns[3])
	n4	=	complex(ns[4])
	n5	=	complex(ns[5])
	#n6	=	complex(ns[6])
	#n6b	=	complex(ns[7])
	n7	=	complex(ns[8])
	g =float(g)/gain_finesse
	
	d2=float(ds[0])
	d3=float(ds[1])
	d4=float(ds[2])
	d5=float(ds[3])
	d6=float(ds[4])
	cls()
	for i in xrange(len(print_array_d)):
		print print_array_d[i],str(float(ds[i]))
	print ''	
	print 'n0= ',n1
	print'n1= ',ns[1]
	print'n2= ',n3
	print'n3= ',n4
	print'n4= ',n5
	print'n5= ',ns[6]
	print'n6= ',n7
	print ''
	print 'g= ',g
	
	Roundtrip=array([])
	Roundtripphase=array([])
	Roundtrip_out_left = array([])
	Roundtrip_out_right = array([])
	Ref_check = array([])
	Q_fac = array([])
	
	if Win.d1d5!=(d2==d6) or Win.d2d4!=(d3==d5):
		print 'WARNING: DISTANCE OPTIONS NOT FULLFILLED'
	if Win.n1n5!=(ns[1]==ns[6]) or Win.n2n4!=(n3==n5):
		print 'WARNING: REFRACTIVE INDEX OPTIONS NOT FULLFILLED'
	if Win.auto_gain or  Win.auto_phase:
		lambdas=array([Win.lambda0*1.0])#-1, Win.lambda0,Win.lambda0+1])
		
	for lam in lambdas:
		lam=float(lam)
		if ns[1]=='AU':
			n2=ref_index(lam,AU_N,AU_K,AU_L)
			#if lam==300 or lam==350 or lam==400 or lam==450 or  lam==500  or lam==600 or lam==700:
			#	print(n2)
		elif ns[1]=='AG':
			n2=ref_index(lam,AG_N,AG_K,AG_L)
			#if lam==300 or lam==350 or lam==400 or lam==450 or  lam==500  or lam==600 or lam==700:
			#	print(n2)
		else:
			n2=complex(ns[1])
		Ref_check=append(Ref_check,n2)
		if ns[6]=='AU':
			n6 = ref_index(lam,AU_N,AU_K,AU_L)	
			#print(n6)
		elif ns[6]=='AG':
			n6 = ref_index(lam,AG_N,AG_K,AG_L)
		else:
			n6=complex(ns[6])
		try:		
			Win.progressBar.setValue(min(100/(lambdas[-1]-lambdas[0])*(lam-lambdas[0]),100))
		except:
			print()
		#print()
		M54	=Mij(n5,n4)
		M4	=Mi(n4,d4,lam,g)
		M43	=Mij(n4,n3)
		M3	=Mi(n3,d3,lam,0)
		M32	=Mij(n3,n2)
		M2	=Mi(n2,d2,lam,0)
		M21	=Mij(n2,n1)
		M5	=Mi(n5,d5,lam,0)
		M56	=Mij(n5,n6)
		M6	=Mi(n6,d6,lam,0)
		M67	=Mij(n6,n7)

		Mtot_right=dot(M67,dot(M6,dot(M56,M5)))
		Stot_right=S(Mtot_right)

		Mtot_left=dot(M21,dot(M2,dot(M32,dot(M3,dot(M43,dot(M4,M54))))))
		Stot_left=S(Mtot_left)

		v_start = array([[1],[0]])
		v_left	= dot(Stot_left,v_start)


		v_start_rigth = array([[v_left[1,0]],[0]])
		v_right =dot(Stot_right,v_start_rigth) 
		
		#---------Auto Section--------------
		
		if int(lam)==int(Win.lambda0):
			angl = angle(v_right[1,0],deg=1)
			ampl = absolute(v_right[1,0])**2
			if Win.auto_gain:
				if abs(ampl-1)<=1e-3:
					Win.auto_gain=False
					return [0,0]
				else:
					return [-1,ampl-1]
			elif Win.auto_phase:
				if absolute(angl)<1e-4:
					Win.auto_phase=False
					return [0,0]
				else:
					return [-2,angl]#/((phase_right-phase_left)/2.0)]
		#---------------------------------			
		

		#Stot_right[1,0]*Stot_left[1,0]
		Roundtrip=append(Roundtrip,absolute(v_right[1,0])**2)
		Roundtripphase = append(Roundtripphase,angle(v_right[1,0],deg=1))
		Roundtrip_out_left =append(Roundtrip_out_left, (n1/n3).real * absolute(v_left[0,0])**2)
		Roundtrip_out_right=append(Roundtrip_out_right,(n7/n3).real * absolute( v_right[0,0])**2)
		Q_fac=append(Q_fac,-4*Pi/(lam)*(n3*d3+n5*d5+n4*d4)/log(Roundtrip[-1]))	
	#plt.figure(2)
	#plt.plot(lambdas,Ref_check.real)
	#plt.show(2)
	return [Roundtrip,Roundtripphase,Roundtrip_out_left,Roundtrip_out_right,Q_fac]
def c_tester(arr):
	for x in arr:
		if x.imag!=0:
			print('element',x,' is complex')

#--------------GUI-------------------
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

def clamp(n, minn=10., maxn=700.):
	
    return max(min(maxn, n), minn)

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def get_complex(string):
	if string=="":
		return ""
	string=str(string)
	string="".join(string.split()) #remove all whitespaces
	R=string[0] #get sign or first digit
	I=''
	got_R=False
	got_I=False
	for char in string[1:]: #skip first digit
		if not(got_R):
			if char.isdigit() or char=='.':
				R+=char
				continue
			elif char=='+' or char=='-':
				got_R=True
				I+=char
		else:
			if char=='j' or char=='i':
				got_I=True
			if char.isdigit() or char=='.':
				I+=char
				continue
	if got_I:
		return complex(R+I+'j')
	else:
		return float(R)
			
class Ui_MainWindow(object):
	def init_rest(self):
		self.PlotCanv() #Canvas
		self.progressBar.setValue(0)
		
		#---
		self.d2d4=False
		self.n1n5=False
		self.d1d5=False
		self.n0n6=False
		self.n2n4=False
		self.auto_gain=False
		self.auto_phase=False
		self.n_array=array([])
		self.d_array=array([])
		self.lambdas=array([])
		self.g=0
		self.lambda0=0
		#self.btnpushed=False
		
		#---res first
		#self.handleButton()
		#--Sliders
		
		self.Slider_1.setTracking(True)
		self.Slider_1.setMaximum(700)
		self.Slider_1.setMinimum(10)
		self.Slider_1.sliderReleased.connect(self.plot1)
		self.Slider_1.sliderMoved.connect(self.update_sldr_lbl)
		self.Slider_1.setDisabled(True)
		
		self.Slider_2.setTracking(True)
		self.Slider_2.setMaximum(700)
		self.Slider_2.setMinimum(10)
		self.Slider_2.sliderReleased.connect(self.plot1)
		self.Slider_2.sliderMoved.connect(self.update_sldr_lbl)
		self.Slider_2.setDisabled(True)
		
		self.Slider_3.setTracking(True)
		self.Slider_3.setMaximum(700)
		self.Slider_3.setMinimum(10)
		self.Slider_3.sliderReleased.connect(self.plot1)
		self.Slider_3.sliderMoved.connect(self.update_sldr_lbl)
		self.Slider_3.setDisabled(True)
		
		self.Slider_4.setTracking(True)
		self.Slider_4.setMaximum(700)
		self.Slider_4.setMinimum(10)
		self.Slider_4.sliderReleased.connect(self.plot1)
		self.Slider_4.sliderMoved.connect(self.update_sldr_lbl)
		self.Slider_4.setDisabled(True)
		
		self.Slider_5.setTracking(True)
		self.Slider_5.setMaximum(700)
		self.Slider_5.setMinimum(10)
		self.Slider_5.sliderReleased.connect(self.plot1)
		self.Slider_5.sliderMoved.connect(self.update_sldr_lbl)
		self.Slider_5.setDisabled(True)
		
		self.Slider_6.setTracking(True)
		self.Slider_6.setMaximum(20000)
		self.Slider_6.setMinimum(-10)
		self.Slider_6.sliderReleased.connect(self.plot1)
		self.Slider_6.sliderMoved.connect(self.update_sldr_lbl)
		#self.Slider_6.valueChanged[int].connect(self.auto_g)
		self.Slider_6.setDisabled(True)
		self.Slider_6.setSingleStep(0.1)
		#--Checkboxes
		self.checkBox_d1d5.toggled.connect(self.bind)
		self.checkBox_d2d4.toggled.connect(self.bind)
		self.checkBox_n1n5.toggled.connect(self.bind)
		self.checkBox_n2n4.toggled.connect(self.bind)
		self.checkBox_n0n6.toggled.connect(self.bind)
		#---Buttons
		self.pushButton.clicked.connect(self.handleButton)
		self.pushButton_auto_g.clicked.connect(self.set_auto)
		self.pushButton_auto_g.setDisabled(True)
		self.pushButton_auto_phase.clicked.connect(self.set_auto)
		self.pushButton_auto_phase.setDisabled(True)
		
		#---combo box
		self.comboBox_LM.currentIndexChanged.connect(self.update_cmbb_lbl)
		self.comboBox_RM.currentIndexChanged.connect(self.update_cmbb_lbl)
		
		#---some values
		self.lineEdit_5.setText('500')
		self.lineEdit_n0.setText('1.87')
		self.lineEdit_n1.setText('')
		self.lineEdit_n2.setText('2.07')
		self.lineEdit_n3.setText('2.07')
		self.lineEdit_5.setText('730')
		self.lineEdit_d1.setText('80')
		self.lineEdit_d2.setText('100')
		self.lineEdit_d3.setText('50')
		self.lineEdit_g.setText('0')
		self.lineEdit_range.setText('300')
		self.checkBox_d1d5.setChecked(False)
		self.checkBox_d1d5.setChecked(True)
		self.checkBox_d2d4.setChecked(True)
		self.checkBox_n1n5.setChecked(True)
		self.checkBox_n2n4.setChecked(True)
		self.checkBox_n0n6.setChecked(True)
		self.comboBox_LM.setCurrentIndex(1)
		self.comboBox_LM.setCurrentIndex(0)
		
		
		#self.comboBox_RM.setCurrentIndex(2)
	def check_free_fields(self):
		if self.comboBox_RM.currentText()=='Bragg' and self.comboBox_LM.currentText()=='Bragg':
			self.lineEdit_d1.setDisabled(True)
			self.lineEdit_d5.setDisabled(True)
		if self.comboBox_RM.currentText()=='Bragg' and self.comboBox_LM.currentText()!='Bragg':
			self.lineEdit_d1.setDisabled(False)
			self.lineEdit_d5.setDisabled(True)
		if self.comboBox_RM.currentText()!='Bragg' and self.comboBox_LM.currentText()=='Bragg':
			self.lineEdit_d1.setDisabled(True)
			self.lineEdit_d5.setDisabled(False)	
		if self.comboBox_RM.currentText()!='Bragg' and self.comboBox_LM.currentText()!='Bragg':
			self.lineEdit_d1.setDisabled(False)
			self.lineEdit_d5.setDisabled(self.d1d5)	
	
	def bind(self):
		s=MainWindow.sender()
		if s.text()=="d2=d4":
			self.lineEdit_d4.setDisabled(s.isChecked())
			self.Slider_4.setDisabled(s.isChecked())
			self.lineEdit_d4.setText('')
			self.d2d4=s.isChecked()
		elif s.text()=="n1=n5":
			self.lineEdit_n5.setText('')
			self.lineEdit_n5.setDisabled(s.isChecked())
			self.lineEdit_n5b.setDisabled(s.isChecked())
			self.comboBox_RM.setDisabled(s.isChecked())
			self.lineEdit_n5b.setDisabled(self.comboBox_RM.currentText()=='Silver' or self.comboBox_RM.currentText()=='Gold')
			self.lineEdit_n5.setDisabled( self.comboBox_RM.currentText()=='Silver' or self.comboBox_RM.currentText()=='Gold')
			self.n1n5=s.isChecked()	
			if self.n1n5:
				self.comboBox_RM.setCurrentIndex(self.comboBox_LM.currentIndex())		
		elif s.text()=="d1=d5":
			self.lineEdit_d5.setText('')
			self.lineEdit_d5.setDisabled(s.isChecked())
			self.Slider_5.setDisabled(s.isChecked())
			self.d1d5=s.isChecked()
		elif s.text()=="n0=n6":
			self.lineEdit_n6.setText('')
			self.lineEdit_n6.setDisabled(s.isChecked())
			self.n0n6=s.isChecked()
		elif s.text()=="n2=n4":
			self.lineEdit_n4.setText('')
			self.lineEdit_n4.setDisabled(s.isChecked())
			self.n2n4=s.isChecked()
		self.check_free_fields()	
	def PlotCanv(self):
		self.figure = plt.figure(figsize=(15,5))
		self.canvas = FigureCanvas(self.figure)
		self.canvas.toolbar = NavigationToolbar(self.canvas,self.canvas,self.canvas)
		self.canvas.setWindowTitle("Plot")
		self.canvas.setGeometry(500, 200, 1000, 1000)
		self.canvas.setObjectName(_fromUtf8("canvas"))
			
		self.canvas.move(QtGui.QDesktopWidget().availableGeometry().topRight())
				
		self.canvas.show()
	
	
	def update_sldr_lbl(self):

		self.label_slider_d1.setText(str(self.Slider_1.value()))
		self.label_slider_d2.setText(str(self.Slider_2.value()))
		self.label_slider_d3.setText(str(self.Slider_3.value()))
		self.label_slider_d4.setText(str(self.Slider_4.value()))
		self.label_slider_d5.setText(str(self.Slider_5.value()))
		self.label_slider_g.setText(str(float(self.Slider_6.value())/gain_finesse))
		
	def update_appear(self):
		Right_Mirror = self.comboBox_RM.currentText()
		Left_Mirror	= self.comboBox_LM.currentText()
		if Left_Mirror != 'Bragg' and Right_Mirror != 'Bragg':
				self.label_18.setPixmap(QtGui.QPixmap(_fromUtf8("metal_setup.png")))
				self.label_n1a.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">1</span></p></body></html>", None))
				self.label_n5a.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">5</span></p></body></html>", None))
		elif Left_Mirror != 'Bragg' and Right_Mirror == 'Bragg':
				self.label_18.setPixmap(QtGui.QPixmap(_fromUtf8("setup_LM_RB.png")))
				self.label_n1a.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">1</span></p></body></html>", None))
				self.label_n5a.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">5a</span></p></body></html>", None))
		elif Left_Mirror == 'Bragg' and Right_Mirror == 'Bragg':
				self.label_18.setPixmap(QtGui.QPixmap(_fromUtf8("setup_LB_RB.png")))
				self.label_n1a.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">1a</span></p></body></html>", None))
				self.label_n5a.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">5a</span></p></body></html>", None))
		elif Left_Mirror == 'Bragg' and Right_Mirror != 'Bragg':
				self.label_18.setPixmap(QtGui.QPixmap(_fromUtf8("setup_LB_RM.png")))
				self.label_n1a.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">1a</span></p></body></html>", None))
				self.label_n5a.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">5</span></p></body></html>", None))
		#self.lineEdit_NL.setDisabled(Left_Mirror != "Bragg Mirror")
		#self.lineEdit_NR.setDisabled(Right_Mirror != "Bragg Mirror")
		
	def update_cmbb_lbl(self,idx):
		s=MainWindow.sender()
		#print(s.objectName(),s.currentText(),idx)
		if s.objectName()== "comboBox_LM":
			self.lineEdit_n1.setDisabled(idx==0 or idx==1)
			self.lineEdit_n1b.setVisible(idx==3)
			self.label_n1b.setVisible(idx==3)
			self.lineEdit_d1.setDisabled(idx==3)
			self.lineEdit_NL.setDisabled(idx!=3)
			self.lineEdit_NR.setDisabled(self.comboBox_RM.currentText()!='Bragg')
			if idx==3:
				self.checkBox_d1d5.setChecked(False)
				self.checkBox_d1d5.setDisabled(True)
				self.lineEdit_d5.setDisabled(True)
			else:
				self.checkBox_d1d5.setDisabled(False)
			if self.n1n5:
				self.comboBox_RM.setCurrentIndex(idx)
				
			#self.lineEdit_n1.setText('')
		if s.objectName()== "comboBox_RM":
			self.lineEdit_n5.setDisabled(self.checkBox_n1n5.isChecked() or idx!=2)	
			self.lineEdit_n5b.setVisible(idx==3)
			self.lineEdit_d5.setDisabled(idx==3 or self.d1d5)			
			self.label_n5b.setVisible(idx==3)
			self.lineEdit_NR.setDisabled( idx!=3)
			self.lineEdit_n5b.setDisabled(self.n1n5 or self.comboBox_RM.currentText()=='Silver' or self.comboBox_RM.currentText()=='Gold')
			self.lineEdit_n5.setDisabled(self.n1n5 or self.comboBox_RM.currentText()=='Silver' or self.comboBox_RM.currentText()=='Gold')
			
			#self.lineEdit_n5.setText('')
		self.update_appear()
		self.check_free_fields()
					
	def Coll_Ref_IDX(self):
		#--Collect Refractive index
		
		N_0=get_complex(self.lineEdit_n0.text()) 
		
		N_1b = -1 #indicates Left Mirror is not a Bragg Mirror
		NL   = -1
		N_5b = -1 #indicates Right Mirror is not a Bragg Mirror
		NR   = -1
		
		if self.comboBox_LM.currentText() == 'Gold':
			N_1='AU'
		elif self.comboBox_LM.currentText() == 'Silver':
			N_1='AG'
		elif self.comboBox_LM.currentText() == 'Bragg':
			N_1=get_complex(self.lineEdit_n1.text())
			N_1b=get_complex(self.lineEdit_n1b.text())
			try:
				NL=clamp(int(self.lineEdit_NL.text()),0,50)
			except ValueError:
				print '!!! '
				print 'Missing valid value from N_L'
				print '!!! '
				return -1
		else:
			N_1=get_complex(self.lineEdit_n1.text())
			
		N_2=get_complex(self.lineEdit_n2.text())
		N_3=get_complex(self.lineEdit_n3.text())
		N_4=get_complex(self.lineEdit_n4.text())
		
		if self.comboBox_RM.currentText() == 'Gold':
			N_5='AU'
		elif self.comboBox_RM.currentText() == 'Silver':
			N_5='AG'
		elif self.comboBox_RM.currentText() == 'Bragg':
			N_5=get_complex(self.lineEdit_n5.text())
			N_5b=get_complex(self.lineEdit_n5b.text())
			try:
				NR=clamp(int(self.lineEdit_NR.text()),0,50)
			except ValueError:
				print '!!! '
				print 'Missing valid value from N_R'
				print '!!! '
				return -1
		else:
			N_5=get_complex(self.lineEdit_n5.text())
		
		N_6=get_complex(self.lineEdit_n6.text())
		
		if self.n1n5: #if n1=n5==True
			N_5=N_1
			N_5b=N_1b
		if self.n0n6:
			N_6=N_0
		if self.n2n4:
			N_4=N_2

		self.n_array=array([N_0,N_1,N_1b,N_2,N_3,N_4,N_5,N_5b,N_6])
		self.NBR_array=array([NL,NR])
		#print(self.n_array)
	
	def Coll_g_lam(self):
		#--Collect Gain
		G=(self.lineEdit_g.text()).toInt()[0]
		self.Slider_6.setValue(G)
		self.label_slider_g.setText(str(G))
		self.g=G
		#--Collect Center Wavelength
		self.lambda0=(self.lineEdit_5.text()).toInt()[0]
		self.range=(self.lineEdit_range.text()).toInt()[0]
		self.range=self.lambda0-clamp(self.lambda0-self.range,100,2000)
		self.range=clamp(self.lambda0+self.range,100,2000)-self.lambda0
		self.lineEdit_range.setText(str(self.range))
		self.lambdas=linspace(self.lambda0-self.range,self.lambda0+self.range,2*self.range)
		#self.lambdas=ndarray.tolist(self.lambdas)
		#if not(self.lambda0 in self.lambdas):
		#	insort_left(self.lambdas, self.lambda0)
		#	self.lambdas=asarray(self.lambdas)
 
		#print('l: ',lambda0)
		
	def Coll_dist(self):
		#--Collect Distances
		D_1=clamp((self.lineEdit_d1.text()).toFloat()[0])
		D_2=clamp((self.lineEdit_d2.text()).toFloat()[0])
		D_3=clamp((self.lineEdit_d3.text()).toFloat()[0])
		D_4=clamp((self.lineEdit_d4.text()).toFloat()[0])
		D_5=clamp((self.lineEdit_d5.text()).toFloat()[0])
		#print(self.d1d5)
		if self.d1d5:
			D_5=D_1
			
		if self.d2d4:
			D_4=D_2
		self.d_array=array([D_1,D_2,D_3,D_4,D_5])
		self.Slider_1.setValue(D_1)
		self.Slider_2.setValue(D_2)
		self.Slider_3.setValue(D_3)
		self.Slider_4.setValue(D_4)
		#self.btnpushed=False
		self.Slider_5.setValue(D_5)
		self.update_sldr_lbl()
		#print('D: ',self.d_array)	
		
	def handleButton(self):
		#self.btnpushed=True
		if self.Coll_Ref_IDX()==-1:
			return
		
		self.Coll_g_lam()
		self.Coll_Ref_IDX()
		self.Coll_dist()
		#activate slider after first time
		self.Slider_1.setDisabled(False)
		self.Slider_2.setDisabled(False)
		self.Slider_3.setDisabled(False)
		self.Slider_4.setDisabled(self.d2d4)
		self.Slider_5.setDisabled(self.d1d5)
		self.Slider_6.setDisabled(False)
		self.pushButton_auto_g.setDisabled(False)
		self.pushButton_auto_phase.setDisabled(False)
		self.plot1()
	
	def bonded_slider_update(self):
		if self.d1d5:
			self.Slider_5.setValue(int(self.Slider_1.value()))
		if self.d2d4:
			self.Slider_4.setValue(int(self.Slider_2.value()))
		
	def set_dist(self):
		D_1=float(self.Slider_1.value())
		D_2=float(self.Slider_2.value())
		D_3=float(self.Slider_3.value())
		D_4=float(self.Slider_4.value())
		D_5=float(self.Slider_5.value())
		
		self.d_array=array([D_1,D_2,D_3,D_4,D_5])
	
		
	def set_g(self):
		self.g=float(self.Slider_6.value())
		
	def set_auto(self):
		s=MainWindow.sender()
		
		b=s.isChecked()
		
		if s.objectName()== "pushButton_auto_g":
			self.auto_gain=b
			self.pushButton_auto_phase.setDisabled(b)
		elif s.objectName()== "pushButton_auto_phase":
			self.auto_phase=b
			self.pushButton_auto_g.setDisabled(b)
		
		self.lineEdit_d4.setDisabled(b or self.checkBox_d2d4.isChecked() )
		self.Slider_4.setDisabled(b or self.checkBox_d2d4.isChecked())
		self.lineEdit_d1.setDisabled(b)
		self.Slider_1.setDisabled(b)
		self.lineEdit_d2.setDisabled(b)
		self.Slider_2.setDisabled(b)
		self.lineEdit_d3.setDisabled(b)
		self.Slider_3.setDisabled(b)
		self.lineEdit_d5.setDisabled(b or self.checkBox_d1d5.isChecked())
		self.Slider_5.setDisabled(b or self.checkBox_d1d5.isChecked())
		self.Slider_6.setDisabled(b)
		self.lineEdit_n1.setDisabled(b)
		self.lineEdit_n1b.setDisabled(b)
		self.lineEdit_n2.setDisabled(b)
		self.lineEdit_n3.setDisabled(b)
		self.lineEdit_n4.setDisabled(b or self.checkBox_n2n4.isChecked())
		self.lineEdit_n5.setDisabled(b or self.checkBox_n1n5.isChecked())
		self.lineEdit_n5b.setDisabled(b or self.checkBox_n1n5.isChecked())
		self.lineEdit_n6.setDisabled(b or self.checkBox_n0n6.isChecked())
		self.lineEdit_n0.setDisabled(b)
		self.lineEdit_g.setDisabled(b)
		self.lineEdit_range.setDisabled(b)
		self.lineEdit_5.setDisabled(b)
		self.lineEdit_NL.setDisabled(b or self.comboBox_LM.currentText()!='Bragg' )
		self.lineEdit_NR.setDisabled(b or self.comboBox_RM.currentText()!='Bragg')
		
		
		if b and s.objectName()== "pushButton_auto_g":
			self.auto_g()
		elif b and s.objectName()== "pushButton_auto_phase":
			self.auto_ph()
		self.check_free_fields()
	def auto_ph(self):
		self.bonded_slider_update()
		self.set_dist()
		#print(self.d_array)
		#quest=raw_input('Do you want to continue? Y/N   ')		
		if not(self.d2d4):
			cls()
			print '!!! WARNING !!!'
			print 'Option d2==d4 is disabled. Only d2 will be changed.'
			quest=raw_input('Do you want to continue? Y/N:  ')
			if not(quest=='Y' or quest=='y'or quest=='J' or quest=='j'):
				self.auto_phase=False
				self.pushButton_auto_phase.setChecked(False)
				self.set_auto()
				return -1
		
		
		#initial		
		init_d=array([self.d_array[0],self.d_array[1]+10,self.d_array[2],self.d_array[3],self.d_array[4]])
		if self.d2d4:
			init_d[3]=init_d[3]+10
		last_y1 = switch(self.n_array,init_d,self.g,self.lambdas,self.NBR_array,self)[1]
		last_d = init_d[1]
		#loop secant
		while self.auto_phase:
			y=switch(self.n_array,self.d_array,self.g,self.lambdas,self.NBR_array,self)
			step = y[1]/((y[1]-last_y1)/(self.d_array[1]-last_d))
			if abs(step)>50: # not too large steps
				print '!!!WARTNING!!!'
				print 'step is larger then 50. Might be ti far off.'
			if y[0]==-2:
				temp_d = self.d_array[1] #temp save	
				self.d_array[1]=self.d_array[1]-step

				if self.d2d4:
					self.d_array[3]=self.d_array[1]
				last_y1 = y[1] #update
				last_d = temp_d  #update
				
			else:
				self.auto_gain=False
		self.Slider_2.setValue(clamp(round(self.d_array[1],1)))
		#self.lineEdit_d2.setText(str(clamp(int(self.d_array[1]))))
		if self.d2d4:
			self.Slider_4.setValue(clamp((self.lineEdit_d2.text()).toFloat()[0]))
		self.update_sldr_lbl()
		self.plot1()
	
	
	def auto_g(self):
		#print(self.auto_gain)
		self.bonded_slider_update()
		self.set_dist()
		self.g=0
		#initial		
		last_g = self.g+1.
		last_y1 = switch(self.n_array,self.d_array,last_g,self.lambdas,self.NBR_array,self)[1]
		
		#loop secant
		while self.auto_gain:
			y=switch(self.n_array,self.d_array,self.g,self.lambdas,self.NBR_array,self)
			step = y[1]/((y[1]-last_y1)/(self.g-last_g))
			#print y[1]+1
			if y[0]==-1:
				temp_g = self.g #temp save	
				self.g=float(self.g)-step
				last_y1 = y[1] #update
				last_g = temp_g  #update
			else:
				self.auto_gain=False
				#self.pushButton_auto_g.setChecked(False)
				#print self.g
				#self.set_auto()
				#print self.g
		
		self.Slider_6.setValue(float(self.g))
		self.update_sldr_lbl()
		self.plot1()
			
	#def set_lambda0(self):
	#	self.lambda0=(self.lineEdit_5.text()).toInt()[0]
	#	self.lambdas=linspace(self.lambda0-200,self.lambda0+300,self.lambda0+1)
		
	def plot1(self):
		#print(value)
		self.bonded_slider_update()
		self.set_dist()
		self.set_g()
		#self.set_lambda0()
		self.update_sldr_lbl()
		
		#x=linspace(0,value,10)
		#y=x**2
		#print(self.n_array,self.d_array,self.g,self.lambdas)
		y=switch(self.n_array,self.d_array,self.g,self.lambdas,self.NBR_array,self)
		xlimits=[self.lambda0-self.range,self.lambda0+self.range]
		gridcolor = '#969696'
		
		if y==-1 or y==-2 or len(y)==0:
			return
		
		
		y0=y[0].real
		y1=y[1].real
		y2=y[2].real
		y3=y[3].real
		y4=y[4].real
		#c_tester(y0)
		#c_tester(y1)
		#c_tester(y2)
		#c_tester(y3)
		#c_tester(y4)
		plt.ion()
		plt.cla()
		self.figure.clf()
		ax0=self.figure.add_subplot(321)
		
		
		plt.xlim(xlimits)
		plt.ylim(min(y0)*1.09,max(max(y0)*1.1,1.1))
		line0, = ax0.plot(self.lambdas,y0,color='r', linewidth=2.0)
		
		ax0.axvline(self.lambda0,color='k', linestyle='--')
		ax0.axhline(1,color='k', linestyle='-',linewidth=1.8)
		ax0.grid(color=gridcolor, linestyle='-', linewidth=0.8)
		#ax0.set_title('Roundtrip')
		plt.ylabel('Roundtrip')
		plt.xlabel('Wavelength (nm)')
		line0.set_ydata(y0)
		self.canvas.draw()
		
		ax1=self.figure.add_subplot(322)
		plt.cla()
		plt.xlim(xlimits)
		plt.ylim(min(y1)-5,max(y1)+5)
		line1, =ax1.plot(self.lambdas,y1,color='r', linewidth=2.0)
		line1.set_ydata(y1)
		ax1.axhline(0,color='k', linestyle='-',linewidth=1.8)
		ax1.axvline(self.lambda0,color='k', linestyle='--')
		ax1.grid(color=gridcolor, linestyle='-', linewidth=0.8)
		plt.ylabel('Roundtripphase')
		plt.xlabel('Wavelength (nm)')
		
		ax2=self.figure.add_subplot(323)
		plt.cla()
		plt.xlim(xlimits)
		plt.ylim(min(y2)*1.09,max(y2)*1.1) 
		line2, =ax2.plot(self.lambdas,y2,color='r', linewidth=2.0)
		line2.set_ydata(y2)
		ax2.axhline(1,color='k', linestyle='--')
		ax2.axvline(self.lambda0,color='k', linestyle='--')
		ax2.grid(color=gridcolor, linestyle='-', linewidth=0.8)
		plt.ylabel('Roundtrip out left')
		plt.xlabel('Wavelength (nm)')
		
		ax3=self.figure.add_subplot(324)
		plt.cla()
		plt.xlim(xlimits)
		plt.ylim(min(y3)*1.09,1.1*max(y3))   
		line3, =ax3.plot(self.lambdas,y3,color='r', linewidth=2.0)
		line3.set_ydata(y3)
		ax3.axhline(1,color='k', linestyle='--')
		ax3.axvline(self.lambda0,color='k', linestyle='--')
		ax3.grid(color=gridcolor, linestyle='-', linewidth=0.8)
		plt.ylabel('Roundtrip out right')
		plt.xlabel('Wavelength (nm)')
		
		plt.tight_layout()
		plt.subplots_adjust(top=0.9)
		self.canvas.draw()
		
		ax4=self.figure.add_subplot(313)
		plt.cla()
		plt.xlim(xlimits)
		plt.ylim(min(y4)*1.09,1.1*max(y4))   
		line4, =ax4.plot(self.lambdas,y4,color='r', linewidth=2.0)
		line4.set_ydata(y4)
		ax4.axhline(1,color='k', linestyle='--')
		ax4.axvline(self.lambda0,color='k', linestyle='--')
		ax4.grid(color=gridcolor, linestyle='-', linewidth=0.8)
		plt.ylabel('Quality Factor')
		plt.xlabel('Wavelength (nm)')
		
		plt.tight_layout()
		plt.subplots_adjust(top=0.9)
		self.canvas.draw()
		
		'''
				
		Roundtrip_out_left,Roundtrip_out_right
		
		plt.cla()
		ax0=self.figure.add_subplot(111)
		plt.ion()
		line0, = ax0.plot(self.lambdas,y0,color='r', linewidth=2.0)
		ax0.axhline(1,color='k', linestyle='--')
		ax0.axvline(self.lambda0,color='k', linestyle='--')
		ax0.set_title('Roundtrip')
		plt.ylabel('Roundtrip')
		plt.xlabel('Wavelength (nm)')
		line0.set_ydata(y0)
		#self.canvas.draw()
		
		
		ax1=self.figure.add_subplot(122)
		plt.tight_layout()
		y1=y[1]
		plt.ion()
		plt.cla()
		print(len(y1),len(self.lambdas))
		line1, = ax1.plot(self.lambdas,y1,color='r', linewidth=2.0)
		ax1.axhline(1,color='k', linestyle='--')
		ax1.axvline(self.lambda0,color='k', linestyle='--')
		plt.ylabel('Roundtripphase')
		plt.xlabel('Wavelength (nm)')
		ax1.set_title('Roundtripphase')
		line1.set_ydata(y1)
		
		
		self.canvas.draw()'''


	def setupUi(self, MainWindow):
		MainWindow.setObjectName(_fromUtf8("MainWindow"))
		MainWindow.resize(586, 842)
		font = QtGui.QFont()
		font.setFamily(_fromUtf8("Ubuntu"))
		font.setPointSize(12)
		MainWindow.setFont(font)
		icon = QtGui.QIcon()
		icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/newPrefix/GUI/icon.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
		MainWindow.setWindowIcon(icon)
		self.centralwidget = QtGui.QWidget(MainWindow)
		self.centralwidget.setStatusTip(_fromUtf8(""))
		self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
		self.label_18 = QtGui.QLabel(self.centralwidget)
		self.label_18.setGeometry(QtCore.QRect(80, 60, 411, 101))
		sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
		self.label_18.setSizePolicy(sizePolicy)
		font = QtGui.QFont()
		font.setKerning(True)
		self.label_18.setFont(font)
		self.label_18.setText(_fromUtf8(""))
		self.label_18.setPixmap(QtGui.QPixmap(_fromUtf8("metal_setup.png")))
		self.label_18.setScaledContents(False)
		self.label_18.setWordWrap(True)
		self.label_18.setObjectName(_fromUtf8("label_18"))
		self.pushButton = QtGui.QPushButton(self.centralwidget)
		self.pushButton.setGeometry(QtCore.QRect(340, 470, 71, 31))
		self.pushButton.setCheckable(False)
		self.pushButton.setObjectName(_fromUtf8("pushButton"))
		self.layoutWidget = QtGui.QWidget(self.centralwidget)
		self.layoutWidget.setGeometry(QtCore.QRect(460, 170, 91, 141))
		self.layoutWidget.setObjectName(_fromUtf8("layoutWidget"))
		self.gridLayout_2 = QtGui.QGridLayout(self.layoutWidget)
		self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
		self.checkBox_n2n4 = QtGui.QCheckBox(self.layoutWidget)
		self.checkBox_n2n4.setObjectName(_fromUtf8("checkBox_n2n4"))
		self.gridLayout_2.addWidget(self.checkBox_n2n4, 3, 0, 1, 1)
		self.checkBox_d2d4 = QtGui.QCheckBox(self.layoutWidget)
		self.checkBox_d2d4.setObjectName(_fromUtf8("checkBox_d2d4"))
		self.gridLayout_2.addWidget(self.checkBox_d2d4, 1, 0, 1, 1)
		self.checkBox_n1n5 = QtGui.QCheckBox(self.layoutWidget)
		self.checkBox_n1n5.setObjectName(_fromUtf8("checkBox_n1n5"))
		self.gridLayout_2.addWidget(self.checkBox_n1n5, 2, 0, 1, 1)
		self.checkBox_d1d5 = QtGui.QCheckBox(self.layoutWidget)
		self.checkBox_d1d5.setObjectName(_fromUtf8("checkBox_d1d5"))
		self.gridLayout_2.addWidget(self.checkBox_d1d5, 0, 0, 1, 1)
		self.checkBox_n0n6 = QtGui.QCheckBox(self.layoutWidget)
		self.checkBox_n0n6.setObjectName(_fromUtf8("checkBox_n0n6"))
		self.gridLayout_2.addWidget(self.checkBox_n0n6, 4, 0, 1, 1)
		self.layoutWidget1 = QtGui.QWidget(self.centralwidget)
		self.layoutWidget1.setGeometry(QtCore.QRect(10, 540, 261, 206))
		self.layoutWidget1.setObjectName(_fromUtf8("layoutWidget1"))
		self.gridLayout_4 = QtGui.QGridLayout(self.layoutWidget1)
		self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
		self.label = QtGui.QLabel(self.layoutWidget1)
		self.label.setFrameShadow(QtGui.QFrame.Raised)
		self.label.setTextFormat(QtCore.Qt.RichText)
		self.label.setObjectName(_fromUtf8("label"))
		self.gridLayout_4.addWidget(self.label, 0, 0, 1, 1)
		self.label_3 = QtGui.QLabel(self.layoutWidget1)
		self.label_3.setTextFormat(QtCore.Qt.RichText)
		self.label_3.setObjectName(_fromUtf8("label_3"))
		self.gridLayout_4.addWidget(self.label_3, 2, 0, 1, 1)
		self.label_slider_d1 = QtGui.QLabel(self.layoutWidget1)
		self.label_slider_d1.setObjectName(_fromUtf8("label_slider_d1"))
		self.gridLayout_4.addWidget(self.label_slider_d1, 0, 3, 1, 1)
		self.Slider_3 = QtGui.QSlider(self.layoutWidget1)
		self.Slider_3.setOrientation(QtCore.Qt.Horizontal)
		self.Slider_3.setObjectName(_fromUtf8("Slider_3"))
		self.gridLayout_4.addWidget(self.Slider_3, 2, 1, 1, 2)
		self.label_slider_g = QtGui.QLabel(self.layoutWidget1)
		self.label_slider_g.setObjectName(_fromUtf8("label_slider_g"))
		self.gridLayout_4.addWidget(self.label_slider_g, 5, 3, 1, 1)
		self.Slider_4 = QtGui.QSlider(self.layoutWidget1)
		self.Slider_4.setOrientation(QtCore.Qt.Horizontal)
		self.Slider_4.setObjectName(_fromUtf8("Slider_4"))
		self.gridLayout_4.addWidget(self.Slider_4, 3, 1, 1, 2)
		self.Slider_6 = QtGui.QSlider(self.layoutWidget1)
		self.Slider_6.setOrientation(QtCore.Qt.Horizontal)
		self.Slider_6.setObjectName(_fromUtf8("Slider_6"))
		self.gridLayout_4.addWidget(self.Slider_6, 5, 2, 1, 1)
		self.label_16 = QtGui.QLabel(self.layoutWidget1)
		self.label_16.setObjectName(_fromUtf8("label_16"))
		self.gridLayout_4.addWidget(self.label_16, 4, 0, 1, 1)
		self.label_slider_d4 = QtGui.QLabel(self.layoutWidget1)
		self.label_slider_d4.setObjectName(_fromUtf8("label_slider_d4"))
		self.gridLayout_4.addWidget(self.label_slider_d4, 3, 3, 1, 1)
		self.label_4 = QtGui.QLabel(self.layoutWidget1)
		self.label_4.setObjectName(_fromUtf8("label_4"))
		self.gridLayout_4.addWidget(self.label_4, 3, 0, 1, 1)
		self.label_2 = QtGui.QLabel(self.layoutWidget1)
		self.label_2.setTextFormat(QtCore.Qt.RichText)
		self.label_2.setObjectName(_fromUtf8("label_2"))
		self.gridLayout_4.addWidget(self.label_2, 1, 0, 1, 1)
		self.label_slider_d5 = QtGui.QLabel(self.layoutWidget1)
		self.label_slider_d5.setObjectName(_fromUtf8("label_slider_d5"))
		self.gridLayout_4.addWidget(self.label_slider_d5, 4, 3, 1, 1)
		self.Slider_2 = QtGui.QSlider(self.layoutWidget1)
		self.Slider_2.setOrientation(QtCore.Qt.Horizontal)
		self.Slider_2.setObjectName(_fromUtf8("Slider_2"))
		self.gridLayout_4.addWidget(self.Slider_2, 1, 1, 1, 2)
		self.label_22 = QtGui.QLabel(self.layoutWidget1)
		self.label_22.setObjectName(_fromUtf8("label_22"))
		self.gridLayout_4.addWidget(self.label_22, 5, 0, 1, 2)
		self.label_slider_d2 = QtGui.QLabel(self.layoutWidget1)
		self.label_slider_d2.setObjectName(_fromUtf8("label_slider_d2"))
		self.gridLayout_4.addWidget(self.label_slider_d2, 1, 3, 1, 1)
		self.label_slider_d3 = QtGui.QLabel(self.layoutWidget1)
		self.label_slider_d3.setObjectName(_fromUtf8("label_slider_d3"))
		self.gridLayout_4.addWidget(self.label_slider_d3, 2, 3, 1, 1)
		self.Slider_5 = QtGui.QSlider(self.layoutWidget1)
		self.Slider_5.setOrientation(QtCore.Qt.Horizontal)
		self.Slider_5.setObjectName(_fromUtf8("Slider_5"))
		self.gridLayout_4.addWidget(self.Slider_5, 4, 1, 1, 2)
		self.Slider_1 = QtGui.QSlider(self.layoutWidget1)
		self.Slider_1.setOrientation(QtCore.Qt.Horizontal)
		self.Slider_1.setObjectName(_fromUtf8("Slider_1"))
		self.gridLayout_4.addWidget(self.Slider_1, 0, 1, 1, 2)
		self.progressBar = QtGui.QProgressBar(self.centralwidget)
		self.progressBar.setGeometry(QtCore.QRect(10, 760, 118, 23))
		self.progressBar.setProperty("value", 24)
		self.progressBar.setObjectName(_fromUtf8("progressBar"))
		self.line = QtGui.QFrame(self.centralwidget)
		self.line.setGeometry(QtCore.QRect(40, 520, 581, 16))
		self.line.setFrameShape(QtGui.QFrame.HLine)
		self.line.setFrameShadow(QtGui.QFrame.Sunken)
		self.line.setObjectName(_fromUtf8("line"))
		self.label_10 = QtGui.QLabel(self.centralwidget)
		self.label_10.setGeometry(QtCore.QRect(360, 720, 221, 20))
		font = QtGui.QFont()
		font.setPointSize(9)
		self.label_10.setFont(font)
		self.label_10.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
		self.label_10.setObjectName(_fromUtf8("label_10"))
		self.pushButton_auto_g = QtGui.QPushButton(self.centralwidget)
		self.pushButton_auto_g.setGeometry(QtCore.QRect(361, 601, 111, 31))
		sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.pushButton_auto_g.sizePolicy().hasHeightForWidth())
		self.pushButton_auto_g.setSizePolicy(sizePolicy)
		self.pushButton_auto_g.setCheckable(True)
		self.pushButton_auto_g.setObjectName(_fromUtf8("pushButton_auto_g"))
		self.layoutWidget2 = QtGui.QWidget(self.centralwidget)
		self.layoutWidget2.setGeometry(QtCore.QRect(20, 0, 371, 71))
		self.layoutWidget2.setObjectName(_fromUtf8("layoutWidget2"))
		self.horizontalLayout = QtGui.QHBoxLayout(self.layoutWidget2)
		self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
		self.label_9 = QtGui.QLabel(self.layoutWidget2)
		self.label_9.setObjectName(_fromUtf8("label_9"))
		self.horizontalLayout.addWidget(self.label_9)
		self.lineEdit_5 = QtGui.QLineEdit(self.layoutWidget2)
		self.lineEdit_5.setObjectName(_fromUtf8("lineEdit_5"))
		self.horizontalLayout.addWidget(self.lineEdit_5)
		self.label_23 = QtGui.QLabel(self.layoutWidget2)
		self.label_23.setObjectName(_fromUtf8("label_23"))
		self.horizontalLayout.addWidget(self.label_23)
		self.lineEdit_range = QtGui.QLineEdit(self.layoutWidget2)
		self.lineEdit_range.setObjectName(_fromUtf8("lineEdit_range"))
		self.horizontalLayout.addWidget(self.lineEdit_range)
		self.layoutWidget3 = QtGui.QWidget(self.centralwidget)
		self.layoutWidget3.setGeometry(QtCore.QRect(280, 170, 174, 234))
		self.layoutWidget3.setObjectName(_fromUtf8("layoutWidget3"))
		self.formLayout = QtGui.QFormLayout(self.layoutWidget3)
		self.formLayout.setObjectName(_fromUtf8("formLayout"))
		self.label_12 = QtGui.QLabel(self.layoutWidget3)
		self.label_12.setTextFormat(QtCore.Qt.RichText)
		self.label_12.setObjectName(_fromUtf8("label_12"))
		self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label_12)
		self.lineEdit_d1 = QtGui.QLineEdit(self.layoutWidget3)
		self.lineEdit_d1.setEnabled(True)
		self.lineEdit_d1.setObjectName(_fromUtf8("lineEdit_d1"))
		self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.lineEdit_d1)
		self.label_13 = QtGui.QLabel(self.layoutWidget3)
		self.label_13.setTextFormat(QtCore.Qt.RichText)
		self.label_13.setObjectName(_fromUtf8("label_13"))
		self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_13)
		self.lineEdit_d2 = QtGui.QLineEdit(self.layoutWidget3)
		self.lineEdit_d2.setEnabled(True)
		self.lineEdit_d2.setObjectName(_fromUtf8("lineEdit_d2"))
		self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.lineEdit_d2)
		self.label_14 = QtGui.QLabel(self.layoutWidget3)
		self.label_14.setObjectName(_fromUtf8("label_14"))
		self.formLayout.setWidget(2, QtGui.QFormLayout.LabelRole, self.label_14)
		self.lineEdit_d3 = QtGui.QLineEdit(self.layoutWidget3)
		self.lineEdit_d3.setEnabled(True)
		self.lineEdit_d3.setObjectName(_fromUtf8("lineEdit_d3"))
		self.formLayout.setWidget(2, QtGui.QFormLayout.FieldRole, self.lineEdit_d3)
		self.label_15 = QtGui.QLabel(self.layoutWidget3)
		self.label_15.setObjectName(_fromUtf8("label_15"))
		self.formLayout.setWidget(3, QtGui.QFormLayout.LabelRole, self.label_15)
		self.lineEdit_d4 = QtGui.QLineEdit(self.layoutWidget3)
		self.lineEdit_d4.setEnabled(True)
		self.lineEdit_d4.setObjectName(_fromUtf8("lineEdit_d4"))
		self.formLayout.setWidget(3, QtGui.QFormLayout.FieldRole, self.lineEdit_d4)
		self.label_20 = QtGui.QLabel(self.layoutWidget3)
		self.label_20.setObjectName(_fromUtf8("label_20"))
		self.formLayout.setWidget(4, QtGui.QFormLayout.LabelRole, self.label_20)
		self.lineEdit_d5 = QtGui.QLineEdit(self.layoutWidget3)
		self.lineEdit_d5.setEnabled(True)
		self.lineEdit_d5.setObjectName(_fromUtf8("lineEdit_d5"))
		self.formLayout.setWidget(4, QtGui.QFormLayout.FieldRole, self.lineEdit_d5)
		self.label_24 = QtGui.QLabel(self.layoutWidget3)
		self.label_24.setObjectName(_fromUtf8("label_24"))
		self.formLayout.setWidget(5, QtGui.QFormLayout.LabelRole, self.label_24)
		self.lineEdit_NL = QtGui.QLineEdit(self.layoutWidget3)
		self.lineEdit_NL.setEnabled(True)
		self.lineEdit_NL.setObjectName(_fromUtf8("lineEdit_NL"))
		self.formLayout.setWidget(5, QtGui.QFormLayout.FieldRole, self.lineEdit_NL)
		self.lineEdit_NR = QtGui.QLineEdit(self.layoutWidget3)
		self.lineEdit_NR.setEnabled(True)
		self.lineEdit_NR.setObjectName(_fromUtf8("lineEdit_NR"))
		self.formLayout.setWidget(6, QtGui.QFormLayout.FieldRole, self.lineEdit_NR)
		self.label_25 = QtGui.QLabel(self.layoutWidget3)
		self.label_25.setObjectName(_fromUtf8("label_25"))
		self.formLayout.setWidget(6, QtGui.QFormLayout.LabelRole, self.label_25)
		self.layoutWidget4 = QtGui.QWidget(self.centralwidget)
		self.layoutWidget4.setGeometry(QtCore.QRect(10, 171, 271, 336))
		self.layoutWidget4.setObjectName(_fromUtf8("layoutWidget4"))
		self.gridLayout = QtGui.QGridLayout(self.layoutWidget4)
		self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
		self.label_11 = QtGui.QLabel(self.layoutWidget4)
		self.label_11.setTextFormat(QtCore.Qt.RichText)
		self.label_11.setObjectName(_fromUtf8("label_11"))
		self.gridLayout.addWidget(self.label_11, 0, 0, 1, 1)
		self.lineEdit_n0 = QtGui.QLineEdit(self.layoutWidget4)
		self.lineEdit_n0.setEnabled(True)
		self.lineEdit_n0.setObjectName(_fromUtf8("lineEdit_n0"))
		self.gridLayout.addWidget(self.lineEdit_n0, 0, 1, 1, 1)
		self.label_n1a = QtGui.QLabel(self.layoutWidget4)
		self.label_n1a.setTextFormat(QtCore.Qt.RichText)
		self.label_n1a.setObjectName(_fromUtf8("label_n1a"))
		self.gridLayout.addWidget(self.label_n1a, 1, 0, 1, 1)
		self.lineEdit_n1 = QtGui.QLineEdit(self.layoutWidget4)
		self.lineEdit_n1.setEnabled(True)
		self.lineEdit_n1.setObjectName(_fromUtf8("lineEdit_n1"))
		self.gridLayout.addWidget(self.lineEdit_n1, 1, 1, 1, 1)
		self.label_n1b = QtGui.QLabel(self.layoutWidget4)
		self.label_n1b.setTextFormat(QtCore.Qt.RichText)
		self.label_n1b.setObjectName(_fromUtf8("label_n1b"))
		self.gridLayout.addWidget(self.label_n1b, 2, 0, 1, 1)
		self.lineEdit_n1b = QtGui.QLineEdit(self.layoutWidget4)
		self.lineEdit_n1b.setEnabled(True)
		self.lineEdit_n1b.setObjectName(_fromUtf8("lineEdit_n1b"))
		self.gridLayout.addWidget(self.lineEdit_n1b, 2, 1, 1, 1)
		self.label_6 = QtGui.QLabel(self.layoutWidget4)
		self.label_6.setObjectName(_fromUtf8("label_6"))
		self.gridLayout.addWidget(self.label_6, 3, 0, 1, 1)
		self.lineEdit_n2 = QtGui.QLineEdit(self.layoutWidget4)
		self.lineEdit_n2.setObjectName(_fromUtf8("lineEdit_n2"))
		self.gridLayout.addWidget(self.lineEdit_n2, 3, 1, 1, 1)
		self.label_7 = QtGui.QLabel(self.layoutWidget4)
		self.label_7.setObjectName(_fromUtf8("label_7"))
		self.gridLayout.addWidget(self.label_7, 4, 0, 1, 1)
		self.lineEdit_n3 = QtGui.QLineEdit(self.layoutWidget4)
		self.lineEdit_n3.setObjectName(_fromUtf8("lineEdit_n3"))
		self.gridLayout.addWidget(self.lineEdit_n3, 4, 1, 1, 1)
		self.label_8 = QtGui.QLabel(self.layoutWidget4)
		self.label_8.setObjectName(_fromUtf8("label_8"))
		self.gridLayout.addWidget(self.label_8, 5, 0, 1, 1)
		self.lineEdit_n4 = QtGui.QLineEdit(self.layoutWidget4)
		self.lineEdit_n4.setObjectName(_fromUtf8("lineEdit_n4"))
		self.gridLayout.addWidget(self.lineEdit_n4, 5, 1, 1, 1)
		self.label_n5a = QtGui.QLabel(self.layoutWidget4)
		self.label_n5a.setObjectName(_fromUtf8("label_n5a"))
		self.gridLayout.addWidget(self.label_n5a, 6, 0, 1, 1)
		self.lineEdit_n5 = QtGui.QLineEdit(self.layoutWidget4)
		self.lineEdit_n5.setObjectName(_fromUtf8("lineEdit_n5"))
		self.gridLayout.addWidget(self.lineEdit_n5, 6, 1, 1, 1)
		self.comboBox_RM = QtGui.QComboBox(self.layoutWidget4)
		self.comboBox_RM.setObjectName(_fromUtf8("comboBox_RM"))
		self.comboBox_RM.addItem(_fromUtf8(""))
		self.comboBox_RM.addItem(_fromUtf8(""))
		self.comboBox_RM.addItem(_fromUtf8(""))
		self.comboBox_RM.addItem(_fromUtf8(""))
		self.gridLayout.addWidget(self.comboBox_RM, 6, 2, 1, 1)
		self.label_n5b = QtGui.QLabel(self.layoutWidget4)
		self.label_n5b.setObjectName(_fromUtf8("label_n5b"))
		self.gridLayout.addWidget(self.label_n5b, 7, 0, 1, 1)
		self.lineEdit_n5b = QtGui.QLineEdit(self.layoutWidget4)
		self.lineEdit_n5b.setObjectName(_fromUtf8("lineEdit_n5b"))
		self.gridLayout.addWidget(self.lineEdit_n5b, 7, 1, 1, 1)
		self.label_19 = QtGui.QLabel(self.layoutWidget4)
		self.label_19.setObjectName(_fromUtf8("label_19"))
		self.gridLayout.addWidget(self.label_19, 8, 0, 1, 1)
		self.lineEdit_n6 = QtGui.QLineEdit(self.layoutWidget4)
		self.lineEdit_n6.setObjectName(_fromUtf8("lineEdit_n6"))
		self.gridLayout.addWidget(self.lineEdit_n6, 8, 1, 1, 1)
		self.label_21 = QtGui.QLabel(self.layoutWidget4)
		self.label_21.setObjectName(_fromUtf8("label_21"))
		self.gridLayout.addWidget(self.label_21, 9, 0, 1, 1)
		self.lineEdit_g = QtGui.QLineEdit(self.layoutWidget4)
		self.lineEdit_g.setObjectName(_fromUtf8("lineEdit_g"))
		self.gridLayout.addWidget(self.lineEdit_g, 9, 1, 1, 1)
		self.comboBox_LM = QtGui.QComboBox(self.layoutWidget4)
		self.comboBox_LM.setObjectName(_fromUtf8("comboBox_LM"))
		self.comboBox_LM.addItem(_fromUtf8(""))
		self.comboBox_LM.addItem(_fromUtf8(""))
		self.comboBox_LM.addItem(_fromUtf8(""))
		self.comboBox_LM.addItem(_fromUtf8(""))
		self.gridLayout.addWidget(self.comboBox_LM, 1, 2, 1, 1)
		self.pushButton_auto_phase = QtGui.QPushButton(self.centralwidget)
		self.pushButton_auto_phase.setGeometry(QtCore.QRect(361, 635, 112, 28))
		sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.pushButton_auto_phase.sizePolicy().hasHeightForWidth())
		self.pushButton_auto_phase.setSizePolicy(sizePolicy)
		self.pushButton_auto_phase.setCheckable(True)
		self.pushButton_auto_phase.setObjectName(_fromUtf8("pushButton_auto_phase"))
		self.layoutWidget.raise_()
		self.layoutWidget.raise_()
		self.layoutWidget.raise_()
		self.label_18.raise_()
		self.layoutWidget.raise_()
		self.layoutWidget.raise_()
		self.pushButton.raise_()
		self.progressBar.raise_()
		self.line.raise_()
		self.label_10.raise_()
		self.pushButton_auto_g.raise_()
		self.pushButton_auto_phase.raise_()
		MainWindow.setCentralWidget(self.centralwidget)
		self.statusbar = QtGui.QStatusBar(MainWindow)
		self.statusbar.setObjectName(_fromUtf8("statusbar"))
		MainWindow.setStatusBar(self.statusbar)
		self.menubar = QtGui.QMenuBar(MainWindow)
		self.menubar.setGeometry(QtCore.QRect(0, 0, 586, 26))
		font = QtGui.QFont()
		font.setFamily(_fromUtf8("Ubuntu"))
		self.menubar.setFont(font)
		self.menubar.setObjectName(_fromUtf8("menubar"))
		MainWindow.setMenuBar(self.menubar)

		self.retranslateUi(MainWindow)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)
		
		self.init_rest()
		

	def retranslateUi(self, MainWindow):
		MainWindow.setWindowTitle(_translate("MainWindow", "Metal Laser", None))
		self.pushButton.setToolTip(_translate("MainWindow", "<html><head/><body><p>Sets the start values for the sliders below and runs the simulation for the given values.</p></body></html>", None))
		self.pushButton.setText(_translate("MainWindow", "Submit", None))
		self.checkBox_n2n4.setToolTip(_translate("MainWindow", "<html><head/><body><p>Option to set n<span style=\" vertical-align:sub;\">2</span>=n<span style=\" vertical-align:sub;\">4</span></p></body></html>", None))
		self.checkBox_n2n4.setText(_translate("MainWindow", "n2=n4", None))
		self.checkBox_d2d4.setToolTip(_translate("MainWindow", "<html><head/><body><p>Option to set d<span style=\" vertical-align:sub;\">2</span>=d<span style=\" vertical-align:sub;\">4</span></p></body></html>", None))
		self.checkBox_d2d4.setText(_translate("MainWindow", "d2=d4", None))
		self.checkBox_n1n5.setToolTip(_translate("MainWindow", "<html><head/><body><p>Option to set n<span style=\" vertical-align:sub;\">1</span>=n<span style=\" vertical-align:sub;\">5</span></p></body></html>", None))
		self.checkBox_n1n5.setText(_translate("MainWindow", "n1=n5", None))
		self.checkBox_d1d5.setToolTip(_translate("MainWindow", "<html><head/><body><p>Option to set d<span style=\" vertical-align:sub;\">1</span>=d<span style=\" vertical-align:sub;\">5</span></p></body></html>", None))
		self.checkBox_d1d5.setText(_translate("MainWindow", "d1=d5", None))
		self.checkBox_n0n6.setToolTip(_translate("MainWindow", "<html><head/><body><p>Option to set n<span style=\" vertical-align:sub;\">0</span>=n<span style=\" vertical-align:sub;\">6</span></p></body></html>", None))
		self.checkBox_n0n6.setText(_translate("MainWindow", "n0=n6", None))
		self.label.setText(_translate("MainWindow", "<html><head/><body><p>d<span style=\" vertical-align:sub;\">1</span></p></body></html>", None))
		self.label_3.setText(_translate("MainWindow", "<html><head/><body><p>d<span style=\" vertical-align:sub;\">3</span></p></body></html>", None))
		self.label_slider_d1.setText(_translate("MainWindow", "123", None))
		self.label_slider_g.setText(_translate("MainWindow", "123", None))
		self.label_16.setText(_translate("MainWindow", "<html><head/><body><p>d<span style=\" vertical-align:sub;\">5</span></p></body></html>", None))
		self.label_slider_d4.setText(_translate("MainWindow", "123", None))
		self.label_4.setText(_translate("MainWindow", "<html><head/><body><p>d<span style=\" vertical-align:sub;\">4</span></p></body></html>", None))
		self.label_2.setText(_translate("MainWindow", "<html><head/><body><p>d<span style=\" vertical-align:sub;\">2</span></p></body></html>", None))
		self.label_slider_d5.setText(_translate("MainWindow", "123", None))
		self.label_22.setText(_translate("MainWindow", "<html><head/><body><p>Gain</p></body></html>", None))
		self.label_slider_d2.setText(_translate("MainWindow", "123", None))
		self.label_slider_d3.setText(_translate("MainWindow", "123", None))
		self.label_10.setText(_translate("MainWindow", "Support: blattert (Blatter Tobias)", None))
		self.pushButton_auto_g.setText(_translate("MainWindow", "Match Gain", None))
		self.label_9.setText(_translate("MainWindow", "Center Wavelength:", None))
		self.label_23.setText(_translate("MainWindow", "Range:", None))
		self.label_12.setText(_translate("MainWindow", "<html><head/><body><p>d<span style=\" vertical-align:sub;\">1</span></p></body></html>", None))
		self.lineEdit_d1.setToolTip(_translate("MainWindow", "<html><head/><body><p>Distance according to the picture. </p></body></html>", None))
		self.label_13.setText(_translate("MainWindow", "<html><head/><body><p>d<span style=\" vertical-align:sub;\">2</span></p></body></html>", None))
		self.lineEdit_d2.setToolTip(_translate("MainWindow", "<html><head/><body><p>Distance according to the picture. </p></body></html>", None))
		self.label_14.setText(_translate("MainWindow", "<html><head/><body><p>d<span style=\" vertical-align:sub;\">3</span></p></body></html>", None))
		self.lineEdit_d3.setToolTip(_translate("MainWindow", "<html><head/><body><p>Distance according to the picture. </p></body></html>", None))
		self.label_15.setText(_translate("MainWindow", "<html><head/><body><p>d<span style=\" vertical-align:sub;\">4</span></p></body></html>", None))
		self.lineEdit_d4.setToolTip(_translate("MainWindow", "<html><head/><body><p>Distance according to the picture. </p></body></html>", None))
		self.label_20.setText(_translate("MainWindow", "<html><head/><body><p>d<span style=\" vertical-align:sub;\">5</span></p></body></html>", None))
		self.lineEdit_d5.setToolTip(_translate("MainWindow", "<html><head/><body><p>Distance according to the picture. </p></body></html>", None))
		self.label_24.setText(_translate("MainWindow", "<html><head/><body><p>N<span style=\" vertical-align:sub;\">L</span></p></body></html>", None))
		self.lineEdit_NL.setToolTip(_translate("MainWindow", "<html><head/><body><p>Distance according to the picture. </p></body></html>", None))
		self.lineEdit_NR.setToolTip(_translate("MainWindow", "<html><head/><body><p>Distance according to the picture. </p></body></html>", None))
		self.label_25.setText(_translate("MainWindow", "<html><head/><body><p>N<span style=\" vertical-align:sub;\">R</span></p></body></html>", None))
		self.label_11.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">0</span></p></body></html>", None))
		self.lineEdit_n0.setToolTip(_translate("MainWindow", "<html><head/><body><p>Refractive Index according to the picture. No arithmetic functions provided. For complex values, use: (sign)(float)(sign)(float)(j or i)</p><p>ex: 1.3-0.4j or 1.3-0.4i gives the same complex number 1.3-0.4i</p><p>ex: 1.3-j0.4 or 1.3-i*0.4 gives 1.3-0.4i</p></body></html>", None))
		self.label_n1a.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">1</span></p></body></html>", None))
		self.lineEdit_n1.setToolTip(_translate("MainWindow", "<html><head/><body><p>Refractive Index according to the picture. No arithmetic functions provided. For complex values, use: (sign)(float)(sign)(float)(j or i)</p><p>ex: 1.3-0.4j or 1.3-0.4i gives the same complex number 1.3-0.4i</p><p>ex: 1.3-j0.4 or 1.3-i*0.4 gives 1.3-0.4i</p></body></html>", None))
		self.label_n1b.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">1b</span></p></body></html>", None))
		self.lineEdit_n1b.setToolTip(_translate("MainWindow", "<html><head/><body><p>Refractive Index according to the picture. No arithmetic functions provided. For complex values, use: (sign)(float)(sign)(float)(j or i)</p><p>ex: 1.3-0.4j or 1.3-0.4i gives the same complex number 1.3-0.4i</p><p>ex: 1.3-j0.4 or 1.3-i*0.4 gives 1.3-0.4i</p></body></html>", None))
		self.label_6.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">2</span></p></body></html>", None))
		self.lineEdit_n2.setToolTip(_translate("MainWindow", "<html><head/><body><p>Refractive Index according to the picture. No arithmetic functions provided. For complex values, use: (sign)(float)(sign)(float)(j or i)</p><p>ex: 1.3-0.4j or 1.3-0.4i gives the same complex number 1.3-0.4i</p><p>ex: 1.3-j0.4 or 1.3-i*0.4 gives 1.3-0.4i</p></body></html>", None))
		self.label_7.setText(_translate("MainWindow", "n<sub>3</sub>", None))
		self.lineEdit_n3.setToolTip(_translate("MainWindow", "<html><head/><body><p>Refractive Index according to the picture. No arithmetic functions provided. For complex values, use: (sign)(float)(sign)(float)(j or i)</p><p>ex: 1.3-0.4j or 1.3-0.4i gives the same complex number 1.3-0.4i</p><p>ex: 1.3-j0.4 or 1.3-i*0.4 gives 1.3-0.4i</p></body></html>", None))
		self.label_8.setText(_translate("MainWindow", "n<sub>4</sub>", None))
		self.lineEdit_n4.setToolTip(_translate("MainWindow", "<html><head/><body><p>Refractive Index according to the picture. No arithmetic functions provided. For complex values, use: (sign)(float)(sign)(float)(j or i)</p><p>ex: 1.3-0.4j or 1.3-0.4i gives the same complex number 1.3-0.4i</p><p>ex: 1.3-j0.4 or 1.3-i*0.4 gives 1.3-0.4i</p></body></html>", None))
		self.label_n5a.setText(_translate("MainWindow", "n<sub>5</sub>", None))
		self.lineEdit_n5.setToolTip(_translate("MainWindow", "<html><head/><body><p>Refractive Index according to the picture. No arithmetic functions provided. For complex values, use: (sign)(float)(sign)(float)(j or i)</p><p>ex: 1.3-0.4j or 1.3-0.4i gives the same complex number 1.3-0.4i</p><p>ex: 1.3-j0.4 or 1.3-i*0.4 gives 1.3-0.4i</p></body></html>", None))
		self.comboBox_RM.setToolTip(_translate("MainWindow", "<html><head/><body><p>Gold and Silver data from <span style=\" font-weight:600;\">Babar and Weaver (2015)  </span>(refractiveindex.info)</p></body></html>", None))
		self.comboBox_RM.setItemText(0, _translate("MainWindow", "Gold", None))
		self.comboBox_RM.setItemText(1, _translate("MainWindow", "Silver", None))
		self.comboBox_RM.setItemText(2, _translate("MainWindow", "Custom", None))
		self.comboBox_RM.setItemText(3, _translate("MainWindow", "Bragg", None))
		self.label_n5b.setText(_translate("MainWindow", "<html><head/><body><p>n<span style=\" vertical-align:sub;\">5b</span></p></body></html>", None))
		self.lineEdit_n5b.setToolTip(_translate("MainWindow", "<html><head/><body><p>Refractive Index according to the picture. No arithmetic functions provided. For complex values, use: (sign)(float)(sign)(float)(j or i)</p><p>ex: 1.3-0.4j or 1.3-0.4i gives the same complex number 1.3-0.4i</p><p>ex: 1.3-j0.4 or 1.3-i*0.4 gives 1.3-0.4i</p></body></html>", None))
		self.label_19.setText(_translate("MainWindow", "n<sub>6</sub>", None))
		self.lineEdit_n6.setToolTip(_translate("MainWindow", "<html><head/><body><p>Refractive Index according to the picture. No arithmetic functions provided. For complex values, use: (sign)(float)(sign)(float)(j or i)</p><p>ex: 1.3-0.4j or 1.3-0.4i gives the same complex number 1.3-0.4i</p><p>ex: 1.3-j0.4 or 1.3-i*0.4 gives 1.3-0.4i</p></body></html>", None))
		self.label_21.setText(_translate("MainWindow", "<html><head/><body><p>Gain</p></body></html>", None))
		self.lineEdit_g.setToolTip(_translate("MainWindow", "<html><head/><body><p>Gain value. Internally shrinked by a factor.</p></body></html>", None))
		self.comboBox_LM.setToolTip(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
		"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
		"p, li { white-space: pre-wrap; }\n"
		"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:12pt; font-weight:400; font-style:normal;\">\n"
		"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Gold and Silver data from <span style=\" font-size:small; font-weight:600;\">Babar and Weaver (2015)  </span>(refractiveindex.info)</p>\n"
		"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"> </p></body></html>", None))
		self.comboBox_LM.setItemText(0, _translate("MainWindow", "Gold", None))
		self.comboBox_LM.setItemText(1, _translate("MainWindow", "Silver", None))
		self.comboBox_LM.setItemText(2, _translate("MainWindow", "Custom", None))
		self.comboBox_LM.setItemText(3, _translate("MainWindow", "Bragg", None))
		self.pushButton_auto_phase.setText(_translate("MainWindow", "Match Phase", None))
import test_rc

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

