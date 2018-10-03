#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:34:58 2018

@author: franfcunha
"""
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import ndimage

os.chdir('/home/franfcunha/Desktop/laser_speckle')

from scipy import signal,ndimage

def phase_correlation(img1,img2):
    
    G_a = np.fft.fft2((img1-np.mean(img1))/np.std(img1))
    G_b = np.fft.fft2((img2-np.mean(img2))/np.std(img2))
    conj_b = np.ma.conjugate(G_b)
    R = G_a*conj_b
    R /= np.absolute(R)
    r = np.fft.ifft2(R).real
    
    return r

def cross_correlation(img1,img2):
    
    G_a=np.fft.fft2((img1-np.mean(img1))/np.std(img1))
    G_b = np.fft.fft2((img2-np.mean(img2))/np.std(img2))
    conj_b = np.ma.conjugate(G_b)
    R = G_a*conj_b
    r= np.fft.ifft2(R).real/(img1.size)
    
    return r

def imagens_contraste(lista_frames):
     L=len(lista_frames)
     lista_C=list()
     lista_C_media=np.zeros((1,L))
     
     imcontrast=np.zeros_like(lista_frames[0],dtype=np.float64)
     #para cada frame
     for idx in np.arange(L):
         print(idx)
         intensidade=lista_frames[idx]
         immean = ndimage.uniform_filter(intensidade, size=5)	
         im2mean = ndimage.uniform_filter(np.square(intensidade,dtype=np.uint64), size=5)
         imcontrast += np.sqrt(np.divide(im2mean,np.square(immean,dtype=np.uint64)) - 1)/len(lista_frames)
         lista_C_media[0,idx]=np.average(imcontrast)
         #lista_C.append(imcontrast)
         
         
     return imcontrast,lista_C_media

contraste_L,media_c_L=imagens_contraste(vista_L)
contraste_R,media_c_R=imagens_contraste(vista_R)

pR=np.zeros_like(media_c_R)
for i in np.arange(len(media_c_R[0,:])):
    pR[0,i]=(1-media_c_R[0,i])/media_c_R[0,i]

p_avg=np.convolve(p[0,:], np.ones((25,))/25, mode='valid')    
p_avgR=np.convolve(pR[0,:], np.ones((25,))/25, mode='valid')    


matplotlib.rc('font', weight='bold')
matplotlib.rc('axes', linewidth=2)
matplotlib.rc('font', size=12)
matplotlib.rc('text', usetex=False)

plt.figure()
pl,=plt.plot(p_avg,'-',label='Perfusão L',color='black')
pr,=plt.plot(p_avgR,'--',label='Perfusão R',color='black')
plt.legend(handles=[pl,pr])
plt.title('Blood perfusion values for left and right cameras',fontweight='bold')
plt.xlabel('#Frames',fontweight='bold')
plt.ylabel('Perfusion (a.u.)',fontweight='bold')

plt.figure()
plt.plot(np.abs(np.subtract(media_movel_cL,media_movel_cR)),color='black')
plt.title('Perfusion absolute difference between cameras',fontweight='bold')
plt.xlabel('#Frames',fontweight='bold')

plt.legend(handles=[pl,pr])
def normalizacao(matriz):
    maximo=np.amax(matriz)
    minimo=np.amin(matriz)
    
    matriz_normalizada=np.uint8(np.divide(np.subtract(matriz,minimo),np.subtract(maximo,minimo))*255)
    
    return matriz_normalizada

for i in np.arange(len(contraste_L1)):
    contraste_L1[i]=normalizacao(contraste_L1[i])
    contraste_R1[i]=normalizacao(contraste_R1[i])
    

def diferenca_LR(matL,matR):
    dif=np.abs(np.subtract(matL,matR))
    return dif

dif_SA=diferenca_LR(SA_l,SA_r)
dif_dynrange=diferenca_LR(dyn_range_L,dyn_range_R)    
dif_fj=diferenca_LR(fj_l,fj_r)
dif_scc=diferenca_LR(scc_L,scc_R)
dif_wavent=diferenca_LR(wavent_L,wavent_R)
dif_wgd=diferenca_LR(wgd_L,wgd_R)
