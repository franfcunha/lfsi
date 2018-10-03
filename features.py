#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 17:27:50 2018

@author: franfcunha
"""

import numpy as np

import os
import dill
import sompy

import skfuzzy as fuzz
from pywt import wavedec

from matplotlib import pyplot as plt

os.chdir('/home/franfcunha/Desktop/laser_speckle/scripts/p_paper/aquisicao_29Junho/')

filename='vistas_video2_maisrois.pkl'

dill.load_session(filename)

# Subtraction Average (SA)


def SA(lista_frames):
    
    N=len(lista_frames)
    
    SA_array=np.zeros_like(lista_frames[0],dtype=np.float64)
    
    i=0
    
    while i<N-1:
        SA_array+=np.abs(np.subtract(lista_frames[i],lista_frames[i+1],dtype=np.int64))
        
        i=i+1
    
    SA_array=SA_array/(N-1)
    
    return SA_array 


def fujii_descriptor(lista_frames):
    
    N=len(lista_frames)
    
    fujii_array=np.zeros_like(lista_frames[0],dtype=np.float64)
    
    i=1
    
    while i<N-1:
        fujii_array+=(np.abs(np.subtract(lista_frames[i],lista_frames[i-1],dtype=np.int64)))/(np.add(lista_frames[i],lista_frames[i-1],dtype=np.uint64))
        
        i=i+1
        
        
    
    return fujii_array


def dynamic_range(lista_frames):
    
    N=len(lista_frames)
    
    dyn_range_array=np.zeros_like(lista_frames[0])
    
    N_array=np.stack(lista_frames)
    
    no_rows=np.shape(dyn_range_array)[0]
    no_cols=np.shape(dyn_range_array)[1]
    
    for i in np.arange(no_rows):
        for j in np.arange(no_cols):
            mini=np.amin(N_array[:,i,j])
            maxi=np.amax(N_array[:,i,j])
            
            dyn_range_array[i,j]=maxi-mini
    
    return dyn_range_array

def scc(lista_frames):
    scc_array=np.zeros_like(lista_frames[0])
    
    N=len(lista_frames)
    
    im1=lista_frames[0]
    im2=lista_frames[1]
    
    dif1=np.abs(np.subtract(im1,im2,dtype=np.int64))
    
    delta=np.average(np.ravel(dif1[dif1>0]))

    i=0
    
    while i<N-1:
        dif=np.abs(np.subtract(lista_frames[i],lista_frames[i+1],dtype=np.int64))
        scc_array[dif>=delta]+=1
        
        i=i+1
    
    scc_array=scc_array/(N-1)
    
    return scc_array

def shannon_wav_ent(lista_frames):
    
    N=len(lista_frames)
    ent_array=np.zeros_like(lista_frames[0],dtype=np.float64)
    
    N_array=np.stack(lista_frames)
    
    no_rows=np.shape(ent_array)[0]
    no_cols=np.shape(ent_array)[1]
    
    for i in np.arange(no_rows):
        for j in np.arange(no_cols):
            time_series=N_array[:,i,j]
            wav_dec=wavedec(time_series,'db2')
            
            energies= [np.sum(np.power(x,2)) for x in wav_dec]
            qi= energies/np.sum(energies)
            
            w,k=0,0
            
            n=len(wav_dec)
            
            while k<n:
                w+=qi[k]*np.log(qi[k])
                
                k=k+1
            w=-w
            ent_array[i,j]=w
    return ent_array
              
        
        
def normaliza_imagem(img):
    
    mini=np.amin(img)
    maxi=np.amax(img)
    
    img_norm=(img-mini)/(maxi-mini)
    img_norm = np.array(img_norm,dtype=np.float32)
    return img_norm

def fuzzy_granularity(lista_frames):
    
    #esta função recebe uma lista de frames, e com base no frame intermédio define 3 funções trapezoidais de pertinência 
    
    stack=np.stack(lista_frames[::25])
    avg_mat=np.mean(stack,axis=0)
    
    bins,cnts=np.unique(avg_mat,return_counts=True)
    
    #considerando 3 níveis : dark, medium and light, definir 5 markpoints igualmente espaçados no domínio do array de contagens
    
    step_size=np.size(lista_frames[0])/5
    
    # somas cumulativas das contagens
    
    h=np.cumsum(cnts)
    
    # definição dos markpoints como indices do vetor h
    
    S1=np.size(np.where(h<=step_size))
    S2=np.size(np.where(h<=2*step_size)) 
    S3=np.size(np.where(h<=3*step_size))
    S4=np.size(np.where(h<=4*step_size))
    
    #conversão dos markpoints como valores de intensidade
    
    I_1=int(bins[S1-1])
    I_2=int(bins[S2-1])
    I_3=int(bins[S3-1])
    I_4=int(bins[S4-1])
    
    

    # definição das 3 funções de pertinência trapezoidais: f_dark,f_med,f_light
    
    f_dark=fuzz.membership.trapmf(np.arange(0,I_2+1),[0,0,I_1,I_2])
    f_med=fuzz.membership.trapmf(np.arange(I_1,I_4+1),[I_1,I_2,I_3,I_4])
    f_light=fuzz.membership.trapmf(np.arange(I_3,256),[I_3,I_4,255,255])
    
    plt.plot(np.arange(0,I_2+1),f_dark)
    plt.plot(np.arange(I_1,I_4+1),f_med)
    plt.plot(np.arange(I_3,256),f_light)
    
    return [I_1,I_2,I_3,I_4]
#limiares=fuzzy_granularity(vista_R2)

def granularity(lista_frames,limiares):
    
    N=len(lista_frames)
    Q=0
    
    for j in np.arange(1,N):
        for i in np.arange(3):
            Q+= seq(i,limiares,lista_frames[j],lista_frames[j-1])
        
    Q= Q/N
    
    return Q
    
def seq(indice_textura,limiares_intensidade,imagem_atual,imagem_anterior):
    
    S=np.zeros_like(imagem_atual)
    
    n_rows=np.shape(imagem_atual)[0]
    n_cols=np.shape(imagem_atual)[1]
    
    # textura 'dark'
    if indice_textura==0:
        #limiar_inf=0
        limiar_superior=limiares_intensidade[1]
        
        for k in np.arange(n_rows):
            for l in np.arange(n_cols):
                if (((imagem_anterior[k,l]>limiar_superior)) and (imagem_atual[k,l] <= limiar_superior)):
                    S[k,l]=1
                else:
                    S[k,l]=0
            
    # textura 'medium'
    elif indice_textura==1:
        
        limiar_inf=limiares_intensidade[0]
        limiar_superior=limiares_intensidade[-1]
        
        
        for k in np.arange(n_rows):
            for l in np.arange(n_cols):
                if ((imagem_anterior[k,l] < limiar_inf) or (imagem_anterior[k,l]>limiar_superior)) and ((imagem_atual[k,l] <= limiar_superior) or (imagem_atual[k,l]>= limiar_inf)):
                    S[k,l]=1
                else:
                    S[k,l]=0
    
    # textura 'light'    
    elif indice_textura==2:
        limiar_inf=limiares_intensidade[-2]
        limiar_superior=255
        
        for k in np.arange(n_rows):
            for l in np.arange(n_cols):
                if ((imagem_anterior[k,l] < limiar_inf) and (imagem_atual[k,l]>=limiar_inf)):
                    S[k,l]=1
                else:
                    S[k,l]=0
        
    return S        

def wgd(lista_frames):
     N=len(lista_frames)
     wgd_array=np.zeros_like(lista_frames[0],dtype=np.float64)
    
     for a in np.arange(N-1):
         for b in np.arange(a+1,N):
             wgd_array+=np.abs(np.subtract(lista_frames[a],lista_frames[b],dtype=np.int64))
     
     return wgd_array
             
    
    
## FEATURES GLOBAIS    

SA_l=SA(vista_L[150:])
#SA_l=normaliza_imagem(SA_l)

SA_r=SA(vista_R[150:])
#SA_r=normaliza_imagem(SA_r)

fj_l=fujii_descriptor(vista_L[150:])
#fj_l=normaliza_imagem(fj_l)

fj_r=fujii_descriptor(vista_R[150:])
#fj_r=normaliza_imagem(fj_r)

dyn_range_L=dynamic_range(vista_L[150:])
#dyn_range_L=normaliza_imagem(dyn_range_L)

dyn_range_R=dynamic_range(vista_R[150:])
#dyn_range_R=normaliza_imagem(dyn_range_R)

scc_L=scc(vista_L[150:])
#scc_L=normaliza_imagem(scc_L)

scc_R=scc(vista_R[150:])
#scc_L=normaliza_imagem(scc_R)

wavent_L=shannon_wav_ent(vista_L[150:])
#wavent_L=normaliza_imagem(wavent_L)

wavent_R=shannon_wav_ent(vista_R[150:])
#wavent_R=normaliza_imagem(wavent_R)
    
## FEATURES OCLUSÃO 
SA_l_oclusao=SA(vista_L[150:275])

SA_r_oclusao=SA(vista_R[150:275])


fj_l_oclusao=fujii_descriptor(vista_L[150:275])

fj_r_oclusao=fujii_descriptor(vista_R[150:275])

dyn_range_L_oclusao=dynamic_range(vista_L[150:275])

dyn_range_R_oclusao=dynamic_range(vista_R[150:275])

scc_L_oclusao=scc(vista_L[150:275])

scc_R_oclusao=scc(vista_R[150:275])

wavent_L_oclusao=shannon_wav_ent(vista_L[150:275])
wavent_R_oclusao=shannon_wav_ent(vista_R[150:275])

# Oclusao Testing data

SA_l_oclusao_T=SA(vista_L[185:210])
SA_r_oclusao_T=SA(vista_R[185:210])

fj_l_oclusao_T=fujii_descriptor(vista_L[185:210])
fj_r_oclusao_T=fujii_descriptor(vista_R[185:210])

dyn_range_L_oclusao_T=dynamic_range(vista_L[185:210])
dyn_range_R_oclusao_T=dynamic_range(vista_R[185:210])

scc_L_oclusao_T=scc(vista_L[185:210])
scc_R_oclusao_T=scc(vista_R[185:210])

wavent_L_oclusao_T=shannon_wav_ent(vista_L[185:210])
wavent_R_oclusao_T=shannon_wav_ent(vista_R[185:210])

## FEATURES HIPEREMIA+RECOVERY [primeiros 18s --> 25*18 --> 450 frames

SA_l_hiper=SA(vista_L[450:575])
SA_r_hiper=SA(vista_R[450:575])

fj_l_hiper=fujii_descriptor(vista_L[450:575])
fj_r_hiper=fujii_descriptor(vista_R[450:575])

dyn_range_L_hiper=dynamic_range(vista_L[450:575])
dyn_range_R_hiper=dynamic_range(vista_R[450:575])

scc_L_hiper=scc(vista_L[450:575])
scc_R_hiper=scc(vista_R[450:575])

wavent_L_hiper=shannon_wav_ent(vista_L[450:575])
wavent_R_hiper=shannon_wav_ent(vista_R[450:575])


# Hiper Testing data

SA_l_hiper_T=SA(vista_L[475:500])
SA_r_hiper_T=SA(vista_R[475:500])

fj_l_hiper_T=fujii_descriptor(vista_L[475:500])
fj_r_hiper_T=fujii_descriptor(vista_R[475:500])

dyn_range_L_hiper_T=dynamic_range(vista_L[475:500])
dyn_range_R_hiper_T=dynamic_range(vista_R[475:500])

scc_L_hiper_T=scc(vista_L[475:500])
scc_R_hiper_T=scc(vista_R[475:500])

wavent_L_hiper_T=shannon_wav_ent(vista_L[475:500])
wavent_R_hiper_T=shannon_wav_ent(vista_R[475:500])


##

SA_l_normed=normaliza_imagem(SA_l)
SA_r_normed=normaliza_imagem(SA_r)

dif_SA=np.abs(np.subtract(SA_l,SA_r))

dif_SA_oclusao=np.abs(np.subtract(SA_l_oclusao,SA_r_oclusao))
dif_fj_oclusao=np.abs(np.subtract(fj_l_oclusao,fj_r_oclusao))
dif_dyn_oclusao=np.abs(np.subtract(dyn_range_L_oclusao,dyn_range_R_oclusao))
dif_scc_oclusao=np.abs(np.subtract(scc_L_oclusao,scc_R_oclusao))
dif_wav_oclusao=np.abs(np.subtract(wavent_L_oclusao,wavent_R_oclusao))


dif_SA_hiper=np.abs(np.subtract(SA_l_hiper,SA_r_hiper))
dif_fj_hiper=np.abs(np.subtract(fj_l_hiper,fj_r_hiper))
dif_dyn_hiper=np.abs(np.subtract(dyn_range_L_hiper,dyn_range_R_hiper))
dif_scc_hiper=np.abs(np.subtract(scc_L_hiper,scc_R_hiper))
dif_wav_hiper=np.abs(np.subtract(wavent_L_hiper,wavent_R_hiper))


fj_l_normed=normaliza_imagem(fj_l)
fj_r_normed=normaliza_imagem(fj_r)
dif_fj=np.abs(np.subtract(fj_l,fj_r))

dyn_l_normed= normaliza_imagem(dyn_range_L)
dyn_r_normed= normaliza_imagem(dyn_range_R)
dif_dyn=np.abs(np.subtract(dyn_range_L,dyn_range_R))

dif_scc=np.abs(np.subtract(scc_L,scc_R))

dif_wavent=np.abs(np.subtract(wavent_L,wavent_R))




os.chdir('/home/franfcunha/Desktop/laser_speckle/scripts/p_paper/')
filename='vistas_braco_grandes_features.pkl'

dill.load_session(filename)

def prepara_features_p_som(lista_features):

    L=len(lista_features)
    
    N=np.size(lista_features[0])    
    features_matrix=np.zeros((N,L))
    
    f=0
    
    while f<L:
        descriptor_1d=np.ravel(lista_features[f])
        features_matrix[:,f]=descriptor_1d
        
        f=f+1
    
    for i in np.arange(np.shape(features_matrix)[1]):
        maximo=np.amax(features_matrix[:,i])
        minimo=np.amin(features_matrix[:,i])
        
        features_matrix[:,i]= np.divide((features_matrix[:,i] - minimo),(maximo-minimo))
                    
    return features_matrix
##        
lista_features_oclusaoL=list((SA_l_oclusao,fj_l_oclusao,dyn_range_L_oclusao,scc_L_oclusao,wavent_L_oclusao))
lista_features_oclusaoR=list((SA_r_oclusao,fj_r_oclusao,dyn_range_R_oclusao,scc_R_oclusao,wavent_R_oclusao))

##
lista_features_hiperL=list((SA_l_hiper,fj_l_hiper,dyn_range_L_hiper,scc_L_hiper,wavent_L_hiper))
lista_features_hiperR=list((SA_r_hiper,fj_r_hiper,dyn_range_R_hiper,scc_R_hiper,wavent_R_hiper))

##
lista_features_hiperL_T=list((SA_l_hiper_T,fj_l_hiper_T,dyn_range_L_hiper_T,scc_L_hiper_T,wavent_L_hiper_T))
lista_features_hiperR_T=list((SA_r_hiper_T,fj_r_hiper_T,dyn_range_R_hiper_T,scc_R_hiper_T,wavent_R_hiper_T))

lista_features_oclusaoL_T=list((SA_l_oclusao_T,fj_l_oclusao_T,dyn_range_L_oclusao_T,scc_L_oclusao_T,wavent_L_oclusao_T))
lista_features_oclusaoR_T=list((SA_r_oclusao_T,fj_r_oclusao_T,dyn_range_R_oclusao_T,scc_R_oclusao_T,wavent_R_oclusao_T))

##
oclusao_L=prepara_features_p_som(lista_features_oclusaoL)
oclusao_R=prepara_features_p_som(lista_features_oclusaoR)

hiper_L=prepara_features_p_som(lista_features_hiperL)
hiper_R=prepara_features_p_som(lista_features_hiperR)

#test data

oclusao_L_T=prepara_features_p_som(lista_features_oclusaoL_T)
oclusao_R_T=prepara_features_p_som(lista_features_oclusaoR_T)

hiper_L_T=prepara_features_p_som(lista_features_hiperL_T)
hiper_R_T=prepara_features_p_som(lista_features_hiperR_T)



L=prepara_features_p_som(lista_features_L)
R=prepara_features_p_som(lista_features_R)

dif=list((dif_SA,dif_fj,dif_dyn,dif_scc,dif_wavent))
dif=prepara_features_p_som(dif)
 
dif_oclusao=list((dif_SA_oclusao,dif_fj_oclusao,dif_dyn_oclusao,dif_scc_oclusao,dif_wav_oclusao))
dif_oclusao=prepara_features_p_som(dif_oclusao)

 
dif_hiper=list((dif_SA_hiper,dif_fj_hiper,dif_dyn_hiper,dif_scc_hiper,dif_wav_hiper))
dif_hiper=prepara_features_p_som(dif_hiper)


       
lista_features_L=list((SA_l,fj_l,dyn_range_L,scc_L,wavent_L,contraste_L))
lista_features_R=list((SA_r,fj_r,dyn_range_R,scc_R,wavent_R,contraste_R))

        
lista_features_L2=list((SA_l,fj_l,dyn_range_L,scc_L,wavent_L))
lista_features_R2=list((SA_r,fj_r,dyn_range_R,scc_R,wavent_R))


def converte_matlab(matriz,nome):
    scipy.io.savemat(nome+'.mat',{str(nome): matriz})

converte_matlab(oclusao_R_T,'oclusaoR_T')    
converte_matlab(oclusao_L_T,'oclusaoL_T')
converte_matlab(hiper_L_T,'hiperR_T')
converte_matlab(hiper_R_T,'hiperL_T')
#converte_matlab(hiperR2,'hiperR')

featuresR=prepara_features_p_som(lista_featuresR)

## SELF ORGANIZED MAP

mapsize=[40,40]
som = sompy.SOMFactory.build(featuresR, mapsize, mask=None, mapshape='planar', lattice='rect', initialization='pca', neighborhood='gaussian', training='batch', name='sompy') 
som.train(n_job=1,train_rough_len=3, train_finetune_len=2,verbose='info')



def extrai_plano_componente(objeto_som,indice_variavel,N_som):
    plano=np.reshape(objeto_som.codebook.matrix[:,indice_variavel],(N_som,N_som))
    
    return plano



from sompy.visualization.hitmap import HitMapView

som.cluster(3)
hits  = HitMapView(10,10,"Clustering",text_size=12)
a=hits.show(som)

from sompy.visualization.mapview import View2D,View2DPacked

v=View2D(10,10,'SOM',text_size=10)
v.show(som,what='codebook',which_dim='all',cmap='jet',col_sz=4,desnormalize=True)

from sompy.visualization.bmuhits import BmuHitsView

vhts  = BmuHitsView(10,10,"Hits Map",text_size=5)
vhts.show(som, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)


u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)
#This is the Umat value
UMAT  = u.build_u_matrix(som, distance=1, row_normalized=False)
#Here you have Umatrix plus its render
UMAT = u.show(som, distance2=1, row_normalized=False, show_data=True, contooor=False, blob=False)


