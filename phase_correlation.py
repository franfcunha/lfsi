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
#def plot_phase_corr(lista1,amostragem,*args):
#    
#   lista1=lista1[::amostragem]
#     
#   if len(args)==0:
#       print('Same View correlation:\n')
#       p_c=np.zeros(len(lista1)-1)
#       #campo=np.zeros((len(lista1)-1,2))
#   
#       for i in np.arange(len(p_c)):
#            print(i)
#            r=phase_correlation(lista1[i],lista1[i+1])
#            maximo_space=np.concatenate(np.where(r==r.max()))
#            p_c[i]=r.max()
#            #campo[i,:]=maximo_space
#            print('Offset:\n')
#            print('y: '+str(maximo_space[0])+'\n')
#            print('x: '+str(maximo_space[1])+'\n')
#            
#   else:
#       print('Inter-view correlation:\n')
#       p_c=np.zeros(len(lista1))
#       #campo=np.zeros((len(lista1),2))
#       lista2=args[0][::amostragem]
#       for i in np.arange(len(p_c)):
#           print(i)
#           r=phase_correlation(lista1[i],lista2[i])
#           maximo_space=np.concatenate(np.where(r==r.max()))
#           #campo[i,:]=maximo_space
#           print('Offset:\n')
#           print('y: '+str(maximo_space[0])+'\n')
#           print('x: '+str(maximo_space[1])+'\n')
#           p_c[i]=r.max()
#   
#
#   plt.plot(np.arange(len(p_c))*(amostragem/25),p_c)
#   plt.xlabel('Time (s)',fontsize=21)    
#   plt.ylabel('Phase correlation',fontsize=21)
#   plt.rc('font', weight='bold')
#   plt.xticks(fontsize=16)
#   plt.yticks(fontsize=16)
#   
#   return p_c
#
#def plot_cross_corr(lista1,amostragem,*args):
#    
#    lista1=lista1[::amostragem]
#    
#    if len(args)==0:
#        delta=amostragem/25
#        print('Same View correlation:\n')
#        print('Amostragem = '+str(delta)+' s')
#        c_c=np.zeros(len(lista1)-1)
#        campo=np.zeros((len(lista1)-1,2))
#    
#        for i in np.arange(len(c_c)):
#            r=cross_correlation(lista1[i],lista1[i+1])
#            maximo_space=np.concatenate(np.where(r==r.max()))
#            c_c[i]=r.max()
#            campo[i,:]=maximo_space
#            print('Offset:\n'+str(i))
#            print('y: '+str(maximo_space[0])+';')
#            print('x: '+str(maximo_space[1])+'\n')
#                   
#    else:
#       print('Inter-view correlation:\n')
#       c_c=np.zeros(len(lista1))
#       campo=np.zeros((len(lista1),2))
#       
#       lista2=args[0][::amostragem]
#       
#       for i in np.arange(len(c_c)):
#           r=cross_correlation(lista1[i],lista2[i])
#           maximo_space=np.concatenate(np.where(r==r.max()))
#           c_c[i]=r.max()
#           campo[i,:]=maximo_space
#           print('Offset:\n')
#           print('y: '+str(maximo_space[0])+';')
#           print('x: '+str(maximo_space[1])+'\n')
#           
#    
#    plt.plot(np.arange(len(c_c))*(amostragem/25),c_c,color='k')
#    plt.xlabel('Time (s)',fontsize=21)    
#    plt.ylabel('Cross-correlation',fontsize=21)
#    plt.rc('font', weight='bold')
#    plt.xticks(fontsize=16)
#    plt.yticks(fontsize=16)
#    
#    return c_c,campo

#
#p_corr_R_contraste=plot_phase_corr(vista_L,1)
#p_corr_LR,campoLR=plot_phase_corr(vista_L,1,vista_R)
## surface para o campo
#
#from mpl_toolkits import mplot3d
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter3D(campoRL[:,1], campoRL[:,0],np.arange(2550)*0.2,color='black')
#ax.set_zlabel('\n\nTime (s)',fontsize=20,fontweight='bold')
#ax.set_xlabel('\n\nHorizontal Offset (px)',fontsize=20,fontweight='bold')
#ax.set_ylabel('\n\nVertical Offset (px)',fontsize=20,fontweight='bold')
#ax.set_zticks([0, 65])
#ax.set_xticks([-0.04, 0, 0.04])
#ax.set_yticks([-0.04, 0, 0.04])
#

#from skimage.exposure import rescale_intensity 
#kernel=np.divide(np.ones((5,5)),15)
#
#
#r1=signal.convolve2d(np.square(image_R),kernel,mode='same') - np.square(signal.convolve2d(image_R,kernel,mode='same'))
#r2 = rescale_intensity(r1, out_range=np.uint8).astype(np.uint8)
#
#
#l1=signal.convolve2d(np.square(image_L),kernel,mode='same') - np.square(signal.convolve2d(image_L,kernel,mode='same'))
#l2 = rescale_intensity(l1, out_range=np.uint8).astype(np.uint8)
