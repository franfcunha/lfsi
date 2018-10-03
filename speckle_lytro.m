
% localização da pasta com ficheiros lytro raw


cd('/home/franfcunha/Desktop/LFToolbox0.4')
LFMatlabPathSetup();

cd('/media/franfcunha/TI31299000B/Users/utilizador/Desktop/Samples/');
LFUtilProcessWhiteImages();

% decode das imagens speckle

LFUtilDecodeLytroFolder();

%correção de cor
DecodeOptions.OptionalTasks = 'ColourCorrect';
LFUtilDecodeLytroFolder([], [], DecodeOptions);


[off_e,off_d]=offsets(LF);

function [off_e,off_d] = offsets(light_field)

% FOR cada linha do array de microlentes, guardar 14 valores de tx e ty,
% relativamente a cada par dado pela vista_i e vista central dessa linha. 15
% linhas x 14 vizinhos x 2 coordenadas

off_e=zeros(15,7,2);
off_d=zeros(15,7,2);

for i=1:15
    center=rgb2gray(squeeze(light_field(i,8,:,:,1:3)));
    for j=1:7
        %vizinhos a esquerda
        pair_l=rgb2gray(squeeze(light_field(i,8-j,:,:,1:3)));
        c=normxcorr2(center,pair_l);
        if max(c(:)) == 0
            disp(j)
            disp('vazio')
            yoffSet= nan;
            xoffSet= nan;
        else
            [ypeak, xpeak] = find(c==max(c(:)));
            yoffSet = ypeak-size(center,1);
            xoffSet = xpeak-size(center,2);
        end
        off_e(i,j,:)= [xoffSet,yoffSet];
        
        %vizinhos a direita
        
        pair_r=rgb2gray(squeeze(light_field(i,8+j,:,:,1:3)));
        c=normxcorr2(center,pair_r);
        if max(c(:)) == 0
            disp(j)
            disp('vazio')
            yoffSet= nan;
            xoffSet= nan;
        else
            [ypeak, xpeak] = find(c==max(c(:)));
            yoffSet = ypeak-size(center,1);
            xoffSet = xpeak-size(center,2);
        end
        off_d(i,j,:)= [xoffSet,yoffSet];
    end
end
end





% à esquerda

    
    