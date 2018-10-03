import cv2, os

import numpy as np

from matplotlib import pyplot as plt

from scipy.ndimage import gaussian_filter,distance_transform_edt,label
from  skimage.filters.rank import entropy
from skimage.morphology import disk

from skimage.filters import threshold_otsu


from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border

#frequência de amostragem utilizada na aquisição dos vídeos

fs=25

os.chdir('/media/franfcunha/TI31299000B/Users/utilizador/Videos/17-05-2018')

video_L='L.m2ts'
video_R='R.m2ts'

cap_L=cv2.VideoCapture(video_L)
cap_R=cv2.VideoCapture(video_R)


# a partir do 25º segundo e considerando um período de 54s

cap_L.set(cv2.CAP_PROP_POS_FRAMES,25*fs - 1)
cap_R.set(cv2.CAP_PROP_POS_FRAMES,25*fs - 1)

# extração do canal vermelho de cada uma das vistas

success, image_R = cap_R.read()
image_R = image_R[:,:,2]
success, image_L = cap_L.read()
image_L = image_L[:,:,2]

# aplicação de um filtro passa-baixo a cada uma das vistas

lpass_l=gaussian_filter(image_L,5)
lpass_R=gaussian_filter(image_R,5)

# determinação do valor threshold que maximiza a variância inter-classes (otsu thresholding) de cada uma das imagens filtradas
otsu_l=threshold_otsu(lpass_l)
otsu_R=threshold_otsu(lpass_R)

# binarização das imagens filtradas
bin_L=(lpass_l>=otsu_l).astype(np.uint8)
bin_R=(lpass_R>=otsu_R).astype(np.uint8)


#utilização de um kernel 9x9 para aplicação da operação morfológica de erosão nas imagens binárias resultantes do procedimento anterior
#kernel = np.ones((9,9),np.uint8)
#erosionL = cv2.erode(bin_L,kernel)
#erosionR = cv2.erode(bin_R,kernel)

# aplicação do filtro de entropia para redução da área iluminada para área iluminada AND speckle objetivo
entropy_L=entropy(image=image_L,selem=disk(5),mask=bin_L)
entropy_R=entropy(image=image_R,selem=disk(5),mask=bin_R)

#binarizaçaõ da imagem de entropia e determinação das regiões onde há speckle objetivo
speckle_L=np.zeros(np.shape(entropy_L))

speckle_L=np.array(~(entropy_L>threshold_otsu(entropy_L[entropy_L!=0])),dtype=np.uint8)
speckle_L[entropy_L==0]=0

speckle_R=np.zeros(np.shape(entropy_R))

speckle_R=np.array(~(entropy_R>threshold_otsu(entropy_R[entropy_R!=0])),dtype=np.uint8)
speckle_R[entropy_R==0]=0

objective_speckleL=clear_border(speckle_L)
objective_speckleR=clear_border(speckle_R)


kernel = np.ones((11,11),np.uint8)
closingL=cv2.morphologyEx(objective_speckleL, cv2.MORPH_CLOSE, kernel)
openingL = cv2.morphologyEx(closingL, cv2.MORPH_OPEN, kernel)

closingR= cv2.morphologyEx(objective_speckleR, cv2.MORPH_CLOSE, kernel)
openingR = cv2.morphologyEx(closingR, cv2.MORPH_OPEN, kernel)

_, contoursL, _ = cv2.findContours(openingL, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
 
_, contoursR, _ = cv2.findContours(openingR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
 

contours_areaL=[cv2.contourArea(i) for i in contoursL]
regiao_objectiveL=np.where(contours_areaL==np.array(contours_areaL).max())[0][0]       


contours_areaR=[cv2.contourArea(i) for i in contoursR]
regiao_objectiveR=np.where(contours_areaR==np.array(contours_areaR).max())[0][0]       

contorno_L=contoursL[regiao_objectiveL]
contorno_R=contoursR[regiao_objectiveR]

imagem_RGB_L=cv2.cvtColor(cap_L.read()[1],cv2.COLOR_BGR2RGB)
imagem_RGB_R=cv2.cvtColor(cap_R.read()[1],cv2.COLOR_BGR2RGB)


cv2.drawContours(imagem_RGB_L,contorno_L,-1,(0,255,0), 2)
cv2.drawContours(imagem_RGB_R,contorno_R,-1,(0,255,0), 2)
# função de segmentação das regiões conjuntas, iluminadas pelo laser

# utilização da técnica de segmentação Watershed para separação de duas regiões (speckle objetivo na pele e speckle objetivo na parede)
#segmentacao_L=segmenta(erosionL,40)
#segmentacao_R=segmenta(erosionR,40)

# bounding box em torno da região de speckle objetivo na pele
#ROI_L=localiza_roi(segmentacao_L,2)
#ROI_R=localiza_roi(segmentacao_R,2)

# execução do algoritmo canny edge detector na bounding box envolvente à ROI de speckle no braço...

#contorno_regiaoL=np.zeros((np.shape(erosionL)))
#contorno_regiaoL[ROI_L.get('ymin'):ROI_L.get('ymax'),ROI_L.get('xmin'):ROI_L.get('xmax')]=cv2.Canny(erosionL[ROI_L.get('ymin'):ROI_L.get('ymax'),ROI_L.get('xmin'):ROI_L.get('xmax')],0,1)

#contorno_regiaoR=np.zeros((np.shape(erosionR)))
#contorno_regiaoR[ROI_R.get('ymin'):ROI_R.get('ymax'),ROI_R.get('xmin'):ROI_R.get('xmax')]=cv2.Canny(erosionR[ROI_R.get('ymin'):ROI_R.get('ymax'),ROI_R.get('xmin'):ROI_R.get('xmax')],0,1)
coordenadas_contorno_da_regiaoL=contoursL[regiao_objectiveL][:,0,:]   
coordenadas_contorno_da_regiaoR=contoursR[regiao_objectiveR][:,0,:]

del contorno_L,contorno_R,contours_areaL,contours_areaR,contoursL,kernel,objective_speckleL,objective_speckleR,success,contoursR,regiao_objectiveL,regiao_objectiveR,closingL,closingR,openingL,openingR,speckle_L,speckle_R,entropy_L,entropy_R,bin_L,bin_R,lpass_l,lpass_R,otsu_l,otsu_R,video_L,video_R,fs

## TENTAR APLICAR O ALGORITMO BASEADO EM CURVATURA

os.chdir('/home/franfcunha/Desktop/fac/tese_scripts/curvature_reg')

from interpcurve import interpcurve, point_u_parametrised_boundary
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelextrema
from scipy import optimize
from math import hypot


boundary_1=interpcurve(650,coordenadas_contorno_da_regiaoL[:,0],coordenadas_contorno_da_regiaoL[:,1])
boundary_2=interpcurve(650,coordenadas_contorno_da_regiaoR[:,0],coordenadas_contorno_da_regiaoR[:,1])

del coordenadas_contorno_da_regiaoL,coordenadas_contorno_da_regiaoR

def css_max_matching(boundary_1,boundary_2,epsilon,s):
    
    # CURVATURA

    Np=np.shape(boundary_1)[0]
    
    sigma=np.array([2,4,8,16])

    X1_convoluido=[None] * len(sigma)
    Y1_convoluido= [None] * len(sigma)
    X2_convoluido= [None] * len(sigma)
    Y2_convoluido= [None] * len(sigma)
    #
    
    for i in np.arange(len(sigma)):
        X1_convoluido[i]=gaussian_filter1d(boundary_1[:,0],sigma[i])
        Y1_convoluido[i]=gaussian_filter1d(boundary_1[:,1],sigma[i])
        X2_convoluido[i]=gaussian_filter1d(boundary_2[:,0],sigma[i])
        Y2_convoluido[i]=gaussian_filter1d(boundary_2[:,1],sigma[i])
    
    curvatures_fixed=[]
    curvatures_moving=[]
    
    for i in np.arange(len(sigma)):
        curvatures_fixed.append(curvatura_por_sigma(sigma[i],sigma,X1_convoluido,Y1_convoluido))
        curvatures_moving.append(curvatura_por_sigma(sigma[i],sigma,X2_convoluido,Y2_convoluido))
    
    # RETORNA LISTA COM N MATRIZES, SENDO QUE EM CADA UMA DESTAS, CADA COLUNA REPRESENTA O PAR {U,K(U)}
    
    maximos_relativos_contorno1=css_maxim_detect(curvatures_fixed)
    maximos_relativos_contorno2=css_maxim_detect(curvatures_moving) 
    
    # neste caso obtem-se os maximos que superam o limiar, extraidos do contorno da imagem HE gerado com sigma= sigmas[0]  

    #perimetros das curvas parametrizas
    perimetro_curva_fixed=cv2.arcLength(np.float32(boundary_1),True)
    perimetro_curva_moving=cv2.arcLength(np.float32(boundary_2),True)

    del boundary_1,boundary_2
    

    #threshold para curvatura = 10/PERIMETRO DO CONTORNO
    threshold_contorno1=epsilon/perimetro_curva_fixed
    threshold_contorno2=epsilon/perimetro_curva_moving   


    maximos_refined_fixa = u_acima_threshold(maximos_relativos_contorno1, lambda x: abs(x) >= threshold_contorno1)
    maximos_refined_mov = u_acima_threshold(maximos_relativos_contorno2, lambda x: abs(x) >= threshold_contorno2)
    
    

    # usando somente os ultimos dois sigmas

    # i : numero de maximos de curvatura no contorno 1 e para um dado sigma
    # j : numero de maximos de curvatura no contorno 2 e para um dado sigma 


    maximos_fixa=list()
    maximos_mov=list()

    for i in np.arange(len(maximos_refined_fixa)):
        for j in np.arange(len(maximos_refined_fixa[i][0])):
            maximos_fixa.append(tuple((maximos_refined_fixa[i][0,j],maximos_refined_fixa[i][2,j],sigma[i])))
    
    # a matriz gerada possui o total de máximos obtidos, sendo cada um deles descrito pelo sua posição u, valor de curvatura em sign(k) e o sigma de onde "provem"
    
    
    for i in np.arange(len(maximos_refined_mov)):
        for j in np.arange(len(maximos_refined_mov[i][0])):
            maximos_mov.append(tuple((maximos_refined_mov[i][0,j],maximos_refined_mov[i][2,j],sigma[i])))
    
    # inicializacao da matriz-custo
    matriz_custo=np.zeros((len(maximos_fixa),len(maximos_mov)))

# para os casos em que os maximos da entrada (i,j) provêm de sigmas distintos atribui-se o maior float possivel
# para evitar o matching. Ou seja, CUSTO MÁX.

# quando resultam de sigmas iguais e com sign de curvatura igual aplica-se a equação 6.
# determina-se s com recurso ao script fit_ellipse...

    for i in np.arange(len(maximos_fixa)):
        for j in np.arange(len(maximos_mov)):
            if maximos_fixa[i][2] != maximos_mov[j][2]:
                matriz_custo[i,j]=1.7976931348623157e308
            elif maximos_fixa[i][1] != maximos_mov[j][1]:
                matriz_custo[i,j]=1.7976931348623157e308
            else:
                matriz_custo[i,j]=np.amin((abs(maximos_fixa[i][0]-maximos_mov[j][0]-s),Np-abs(maximos_fixa[i][0]-maximos_mov[j][0]-s)))
    
    
    # novo tamanho para a matriz quadrada... |AB|= |A|+|B|
    size_new_matrix=len(maximos_fixa)+len(maximos_mov)

    # inicialização da nova matriz-custo quadrada
    matriz_custo_2=np.zeros((size_new_matrix,size_new_matrix))

    # preserva-se o valor de C(i,j) nos casos em que i e j estão dentro dos seus limites para aquelas que são
    # as dimensões dos vetores que guardam i maximos da fixa e j maximos da moving

    # aplica-se uma penalização às entradas que caem fora destes limites

    for i in np.arange(size_new_matrix):
        for j in np.arange(size_new_matrix):
            if i < len(maximos_fixa) and j < len(maximos_mov):
                matriz_custo_2[i,j]=matriz_custo[i,j]
            else:
                matriz_custo_2[i,j]=Np/epsilon
                
    
    # a funcao seguinte retorna dois vetores. um com os indices de linhas, e outro com as respetivas posicoes das colunas que emparelham com cada linha
    # ou seja, matching (linha,coluna)
    assignment_hungaro=optimize.linear_sum_assignment(matriz_custo_2)

    # inicializar matriz de zeros com a mesma dimensão da matriz de custo
    hungarian=np.zeros((np.shape(matriz_custo_2)))

    # colocar 1s nas posicoes de matching

    for i in np.arange(len(assignment_hungaro[0])):
        match=assignment_hungaro[1][i]
        hungarian[i,match]=1

    # matriz match_cost é descrita como a matriz_custo_2 mas será "cortada":
     # - guardam-se os valores relativos a entradas (i,j) que o algoritmo hungaro devolveu como 1
     # - penalizam-se máximos sem matching (equação 10 do artigo)
     
    match_cost=matriz_custo_2
     
    for i in np.arange(np.shape(match_cost)[0]):
        for j in np.arange(np.shape(match_cost)[1]):
            if hungarian[i,j]==0:
                match_cost[i,j]=0
            else:
                if hungarian[i,j]==1 and i<len(maximos_fixa) and j<len(maximos_mov):
                    match_cost[i,j]=matriz_custo_2[i,j]
                elif i<len(maximos_fixa) and j>=len(maximos_mov):
                    match_cost[i,j]=Np/5
                elif i>=len(maximos_fixa) and j<len(maximos_mov):
                    match_cost[i,j]=Np/5
                elif i>=len(maximos_fixa) and j>=len(maximos_mov):
                    match_cost[i,j]=0
          
    CUSTO_TOTAL=np.sum(match_cost)
    print('Custo total: '+str(CUSTO_TOTAL)+', com s = '+str(s))
    
    
    # inicializar vetor de matchings
    matches = np.zeros((len(maximos_fixa),2))

    # encontrar os matchings

    for i in np.arange(len(matches)):
        print(i)
        try:
            matches[i,:]=[i,np.nonzero(match_cost[i,:])[0][0]]
        except:
            pass
    
    matches=np.delete(matches,np.where(~matches.any(axis=1))[0],axis=0)
    

    # excluir matchings entre pontos da fixa que combinem com um j que cai fora do numero de maximos de mov
    matches_to_exclude=np.nonzero(matches[:,1]>=len(maximos_mov))
    final_matches=np.delete(matches,matches_to_exclude[0],axis=0)
    final_matches=final_matches.astype(np.int64)
    
    

    pontos_fixa=[maximos_fixa[i] for i in final_matches[:,0]]
    pontos_mov=[maximos_mov[i] for i in final_matches[:,1]]

    matriz_correspondencia=np.zeros((len(pontos_fixa),4))

    for i in np.arange(len(pontos_fixa)):
        indice_sigma_i=np.where(sigma==pontos_fixa[i][2])[0][0]
        matriz_correspondencia[i,0:2]=X1_convoluido[indice_sigma_i][int(pontos_fixa[i][0])],Y1_convoluido[indice_sigma_i][int(pontos_fixa[i][0])]
        matriz_correspondencia[i,2:4]=X2_convoluido[indice_sigma_i][int(pontos_mov[i][0])],Y2_convoluido[indice_sigma_i][int(pontos_mov[i][0])]

    P_fixo=list()
    P_mov=list()     

    for i in np.arange(len(pontos_fixa)):
        P_fixo.append(tuple(np.int64(matriz_correspondencia[i,0:2])))
        P_mov.append(tuple(np.int64(matriz_correspondencia[i,2:4])))


    fix=np.array(P_fixo).reshape(1,-1,2)[0,:,:]
    mov=np.array(P_mov).reshape(1,-1,2)[0,:,:]
    
    return fix,mov

# Funções auxiliares à execução da anterior -------------------------------------------------------------------------------

fix,mov=css_max_matching(boundary_1,boundary_2,.1,.5)

def curvatura_por_sigma(valor_sigma,vetor_sigmas,convolucoes_x,convolucoes_y):
    idx=np.where(vetor_sigmas==valor_sigma)[0]
    
    convolucoes_x=np.array(convolucoes_x)
    convolucoes_y=np.array(convolucoes_y)
    
    x=np.array(convolucoes_x[idx,:])
    y=np.array(convolucoes_y[idx,:])
    
    #implementação da função curvatura k do artigo "Fast method for approximate registration of WSI of serial sections using local curvature"    
    k=np.divide(np.add(np.multiply(gaussian_filter1d(x,sigma=valor_sigma,order=1),gaussian_filter1d(y,sigma=valor_sigma,order=2)),np.multiply(gaussian_filter1d(y,sigma=valor_sigma,order=1),gaussian_filter1d(x,sigma=valor_sigma,order=2))),np.power(np.add(np.power(gaussian_filter1d(x,sigma=valor_sigma,order=1),2),np.power(gaussian_filter1d(y,sigma=valor_sigma,order=1),2)),1.5))
    
    return k

def css_maxim_detect(lista_curvaturas):
    maximos_locais_absolutos=list()
    
    N_sigmas=len(lista_curvaturas)
    
    for i in np.arange(N_sigmas):
        u_maxs=argrelextrema(np.array(lista_curvaturas[i]), np.greater,axis=1)[1]
        u_mins=argrelextrema(np.array(lista_curvaturas[i]),np.less,axis=1)[1]
        u_maximos_locais=np.concatenate((u_maxs,u_mins))
         
         
        #k_u=lista_curvaturas[i][argrelextrema(x, np.greater)[0]]
        k_u=lista_curvaturas[i][0][u_maximos_locais]
                  
        maximos_locais_absolutos.append(np.vstack((u_maximos_locais,k_u)))

    return maximos_locais_absolutos
        
def u_acima_threshold(maximos_relativos,func):
    
    maximos_relativos_refined=list()
    
    for j in np.arange(len(maximos_relativos)):
        
        idxs= [i for (i, val) in enumerate(maximos_relativos[j][1]) if func(val)]
        maximos_relativos_refined.append(np.vstack((maximos_relativos[j][0][idxs],maximos_relativos[j][1][idxs],np.sign(maximos_relativos[j][1][idxs]))))
    
    return maximos_relativos_refined
 

#del X1_convoluido,X2_convoluido, Y1_convoluido, Y2_convoluido, assignment_hungaro, bin_L, bin_R,closingL,closingR,contorno_L,contorno_R,contours_areaL,contours_areaR,contoursL,contoursR,coordenadas_contorno_da_regiaoL,coordenadas_contorno_da_regiaoR,curvatures_fixed,curvatures_moving,entropy_L,entropy_R,final_matches,fs,hungarian,i,image_L,image_R,indice_sigma_i,j,kernel,lpass_l,lpass_R,match,match_cost,matches,matches_to_exclude,matriz_correspondencia,matriz_custo,matriz_custo_2,maximos_fixa,maximos_mov,maximos_refined_fixa,maximos_refined_fixa_3sigmas,maximos_refined_mov,maximos_refined_mov_3sigmas,maximos_relativos_contorno1,maximos_relativos_contorno2,objective_speckleL,objective_speckleR,openingL,openingR,otsu_R,otsu_l,perimetro_curva_fixed,perimetro_curva_moving,pontos_fixa,pontos_mov,regiao_objectiveL,regiao_objectiveR,sigma,size_new_matrix,speckle_L,speckle_R,success,threshold_contorno1,threshold_contorno2,threshold_otsu,video_L,video_R


# TUNING DOS PARÂMETROS
# Pontos definidos nas imagens speckle como referência


''' pontos manualmente identificados na imagem esquerda:
    
    {y,x}
    
    P1-->(441,930)
    P2-->(436,1179)
    P3-->(865,1185)
    
    

    
pontos manualmente identificados na imagem direita:
    
    P1-->(440,641)
    P2-->(435,903)
    P3-->(864,911)

'''

pontos1=np.array([[930,441],[1179,436],[1185,865]])
pontos2=np.array([[641,440],[435,903],[911,864]])
epsilon=np.linspace(0.1,2,20)
s=np.linspace(0,20,40)

def testa_transformacao(pontos1,pontos2,array_epsilon,array_s):
    
    testados=list()
    
    copia1=np.copy(pontos1)
    copia2=np.copy(pontos2)
    
    Np=len(pontos1)
    
    for i in np.arange(len(array_epsilon)):
        for j in np.arange(len(array_s)):
            fix,mov=css_max_matching(boundary_1,boundary_2,array_epsilon[i],array_s[j])
                        
            R,t=rigid_transform_2D(fix,mov)
            t=np.reshape(np.array([t[0,0],t[1,1]]),(2,1))
            transformacao=np.hstack((R,t))
            
            d_med=0
            for k in np.arange(Np):
                #pdb.set_trace()
                ponto_para_modificar=pontos1[k,:]
                pk=transform(ponto_para_modificar,transformacao)
                d_med+=np.linalg.norm(pk-pontos2[k,:])
                
            d_med=d_med/Np
            
            testados.append(d_med)
            
            pontos1=copia1
            pontos2=copia2
            
    return testados
            
            
erros_medios=testa_transformacao(pontos1,pontos2,epsilon,s)
    
def rigid_transform_2D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.matmul(AA.T,BB)

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
       print("Reflection detected")
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    print(t)

    return R, t 


def transform(ponto, transformacao):
  p = np.array([0,0,1])
  for i in range (0,len(ponto)-1):
      p[i] = ponto[i]
  p=np.dot(transformacao,np.transpose(p))
  for i in range (0,len(ponto)-1):
      ponto[i]=p[i]
  return ponto

matrix_copy = [list(row) for row in pontos1]
