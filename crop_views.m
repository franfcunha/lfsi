%% aplica crop a cada uma das vistas após determinar manualmente a região de interesse na vista central de cada linha

function [views_cropped] = crop_views(offset_e,offset_d,LF)

offset=horzcat(fliplr(offset_e),offset_d);
%inicializar cell matrix para guardar crops
views_cropped=cell(15,15);

% para cada linha
for linha=1:15
    % vista central
    center=squeeze(LF(linha,8,:,:,1:3));
    %mostrar
    imshow(center);
    
    %definir rectangulo
    rect=getrect;
    %disp(size(rect));
    
    %arredondamento
    rect=ceil(rect);
    disp(rect)
    
    % aplicar crop
    crop=center(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3));
    views_cropped{linha,8}=crop;
    % para cada linha do MLA
    for col=1:7
    
    % obter coordenadas de offset associadas à posição j
    % relativamente à vista cental desta linha
    
    xoffset=offset(linha,col,1);
    yoffset=offset(linha,col,2);
            
    %se canto
    if( isnan(xoffset) || isnan(yoffset))
                
    else
    % aplicar o offset calculado ao retângulo definido na vista central desta linha e gravar crop (linha,col) na cell matrix    
      crop=rgb2gray(squeeze(LF(linha,col,:,:,1:3)));
     
      views_cropped{linha,col}=crop(rect(2)+yoffset:rect(2)+rect(4)+yoffset,rect(1)+xoffset:rect(1)+rect(3)+xoffset);
    end
       
    end
    
    for col=8:14
    % obter coordenadas de offset associadas à posição j
    % relativamente à vista cental desta linha
    
    xoffset=offset(linha,col,1);
    yoffset=offset(linha,col,2);
            
    %se canto
    if( isnan(xoffset) || isnan(yoffset))
                
    else
    % aplicar o offset calculado ao retângulo definido na vista central desta linha e gravar crop (linha,col) na cell matrix    
      crop=rgb2gray(squeeze(LF(linha,col+1,:,:,1:3)));
     
      views_cropped{linha,col+1}=crop(rect(2)+yoffset:rect(2)+rect(4)+yoffset,rect(1)+xoffset:rect(1)+rect(3)+xoffset);
    end
    end
    
    
        
    end
end

