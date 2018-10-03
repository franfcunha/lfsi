#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:16:55 2018

@author: franfcunha
"""

import os,dill

os.chdir('/home/franfcunha/Desktop/laser_speckle/scripts/p_paper')

dill.load_session('features_braco_reg.pkl')


'''
DETERMINAÇÃO DE QUAL A MÉTRICA MAIS SENSÍVEL AO ÂNGULO DE AQUISIÇÃO
    (p/tal determinar o plano de componente/variável cuja similaridade entre vistas é menor)
'''

oclusaoL_features=list((SA_l_oclusao,dyn_range_L_oclusao,fj_l_oclusao,gran_fuzzyL_oclusao))
features_oclusaoL=prepara_features_p_som(oclusaoL_features)

hiperL_features=list((SA_l_hiper,dyn_range_L_hiper,fj_l_hiper,gran_fuzzyL_hiper))
features_hiperL=prepara_features_p_som(hiperL_features)


oclusaoR_features=list((SA_r_oclusao,dyn_range_R_oclusao,fj_r_oclusao,gran_fuzzyR_oclusao))
features_oclusaoR=prepara_features_p_som(oclusaoR_features)


hiperR_features=list((SA_r_hiper,dyn_range_R_hiper,fj_r_hiper,gran_fuzzyR_hiper))
features_hiperR=prepara_features_p_som(hiperR_features)


mapsize=[40,40]

som_oclusaoL = sompy.SOMFactory.build(features_oclusaoL, mapsize, mask=None, mapshape='planar', lattice='rect', initialization='pca', neighborhood='gaussian', training='batch') 
som_oclusaoL.train(n_job=1,train_rough_len=3, train_finetune_len=2,verbose='info')

som_oclusaoR = sompy.SOMFactory.build(features_oclusaoR, mapsize, mask=None, mapshape='planar', lattice='rect', initialization='pca', neighborhood='gaussian', training='batch') 
som_oclusaoR.train(n_job=1,train_rough_len=3, train_finetune_len=2,verbose='info')


som_hiperL = sompy.SOMFactory.build(features_hiperL, mapsize, mask=None, mapshape='planar', lattice='rect', initialization='pca', neighborhood='gaussian', training='batch') 
som_hiperL.train(n_job=1,train_rough_len=3, train_finetune_len=2,verbose='info')

som_hiperR = sompy.SOMFactory.build(features_hiperR, mapsize, mask=None, mapshape='planar', lattice='rect', initialization='pca', neighborhood='gaussian', training='batch') 
som_hiperR.train(n_job=1,train_rough_len=3, train_finetune_len=2,verbose='info')

#OCLUSAO

sa_som_oclusaoL=extrai_plano_componente(som_oclusaoL,0,40)
sa_som_oclusaoR=extrai_plano_componente(som_oclusaoR,0,40)

dyn_som_oclusaoL=extrai_plano_componente(som_oclusaoL,1,40)
dyn_som_oclusaoR=extrai_plano_componente(som_oclusaoR,1,40)

fj_som_oclusaoL=extrai_plano_componente(som_oclusaoL,2,40)
fj_som_oclusaoR=extrai_plano_componente(som_oclusaoR,2,40)

fz_som_oclusaoL=extrai_plano_componente(som_oclusaoL,3,40)
fz_som_oclusaoR=extrai_plano_componente(som_oclusaoR,3,40)

#RECOVERY


sa_som_hiperL=extrai_plano_componente(som_hiperL,0,40)
sa_som_hiperR=extrai_plano_componente(som_hiperR,0,40)

dyn_som_hiperL=extrai_plano_componente(som_hiperL,1,40)
dyn_som_hiperR=extrai_plano_componente(som_hiperR,1,40)

fj_som_hiperL=extrai_plano_componente(som_hiperL,2,40)
fj_som_hiperR=extrai_plano_componente(som_hiperR,2,40)

fz_som_hiperL=extrai_plano_componente(som_hiperL,3,40)
fz_som_hiperR=extrai_plano_componente(som_hiperR,3,40)



def correlacao_componentes(A,B):
    R=np.corrcoef(np.ravel(A),np.ravel(B))
    
    return R

'''
CALCULANDO AS DIFERENÇAS DE CADA MÉTRICA, ENTRE VISTAS, CONSIDERANDO 2 MOMENTOS FISIOLÓGICOS DISTINTOS (OCLUSÃO VS RECOVERY)
    (para determinar qual a métrica que apresenta maior capacidade de discriminação/ que mais responde ao período de recovery) 
'''

dif_SA_oclusao=np.abs(np.subtract(SA_l_oclusao,SA_r_oclusao))
dif_SA_hiper=np.abs(np.subtract(SA_l_hiper,SA_r_hiper))

dif_dyn_oclusao=np.abs(np.subtract(dyn_range_L_oclusao,dyn_range_R_oclusao))
dif_dyn_hiper=np.abs(np.subtract(dyn_range_L_hiper,dyn_range_R_hiper))

dif_fj_oclusao=np.abs(np.subtract(fj_l_oclusao,fj_r_oclusao))
dif_fj_hiper=np.abs(np.subtract(fj_l_hiper,fj_r_hiper))

dif_granfuzzy_oclusao=np.abs(np.subtract(gran_fuzzyL_oclusao,gran_fuzzyR_oclusao))
dif_granfuzzy_hiper=np.abs(np.subtract(gran_fuzzyL_hiper,gran_fuzzyR_hiper))

dif_oclusao_features=list((dif_SA_oclusao,dif_dyn_oclusao,dif_fj_oclusao,dif_granfuzzy_oclusao))
dif_hiper_features=list((dif_SA_hiper,dif_dyn_hiper,dif_fj_hiper,dif_granfuzzy_hiper))

features_dif_oclusao=prepara_features_p_som(dif_oclusao_features)
features_dif_hiper=prepara_features_p_som(dif_hiper_features)

## SELF ORGANIZED MAP

mapsize=[40,40]

som_oclusao = sompy.SOMFactory.build(features_dif_oclusao, mapsize, mask=None, mapshape='planar', lattice='rect', initialization='pca', neighborhood='gaussian', training='batch') 
som_oclusao.train(n_job=1,train_rough_len=3, train_finetune_len=2,verbose='info')

som_recov = sompy.SOMFactory.build(features_dif_hiper, mapsize, mask=None, mapshape='planar', lattice='rect', initialization='pca', neighborhood='gaussian', training='batch') 
som_recov.train(n_job=1,train_rough_len=3, train_finetune_len=2,verbose='info')

sa_som_oclusao=extrai_plano_componente(som,0,40)
sa_som_recov=extrai_plano_componente(som_recov,0,40)

dyn_som_oclusao=extrai_plano_componente(som,1,40)
dyn_som_recov=extrai_plano_componente(som_recov,1,40)


from sompy.visualization.mapview import View2D,View2DPacked
from matplotlib import pyplot as plt

v1=View2D(100,100,"Data Map",text_size=14)
v1.show(som,col_sz=4,desnormalize=True)


v2=View2D(10,10,'SOM',text_size=10)
v2.show(som_recov,what='codebook',which_dim='all',cmap='jet',col_sz=4,desnormalize=True)