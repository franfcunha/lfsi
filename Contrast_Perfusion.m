

function varargout=Contrast_Perfusion(dados, f_kernel,mediamovel,varargin)

size3D=size(dados,3);

szvarargin=length(varargin);

        for in=1:size3D
            
            A=double(dados(:,:,in));  
            kernel = ones(f_kernel,f_kernel);% These dimensions are arbitrary
            Nk = sum(kernel(:));% soma dos elementos do filtro de kernel
            mu_img = filter2(kernel,A,'valid')/Nk;
            img_sq = filter2(kernel,A.^2,'valid');
            sig_img = sqrt( (img_sq - Nk*mu_img.^2)/(Nk-1));
            C = (sig_img+0.000000000001)./(mu_img+0.000000000001);% local contrast
            m_contraste(:,:,in)=C;
            ContrasteMedio(in)=mean2(C);
        
                 
        end 

if szvarargin==0
    
   varargout{1}=ContrasteMedio;
    
end




for i=1:szvarargin

if strcmpi (varargin{i}, 'contrast')
           

      varargout{i}=ContrasteMedio;
      
      %% Gráfico Contraste:
        figure;
        plot (ContrasteMedio)
        simple = tsmovavg(ContrasteMedio','s',mediamovel,1);
        hold on;
        plot(simple);
        grid on; grid minor;
        xlabel('# Frames');
        ylabel('$$K = {\sigma_{I}\over \bar{I}}$$','Interpreter', 'Latex');
      

elseif strcmpi (varargin{i}, 'perfusion')   
     
    
    for in=1:size3D
           
            P=(1-m_contraste(:,:,in))./m_contraste(:,:,in);
            m_perfusao (:,:,in)=P;
            PerfusaoMedia(in)=nanmean(P(:));
                 
        end 
   
      varargout{i}=PerfusaoMedia;
       %% Gráfico Perfusão: 
        
        figure; 
        plot (PerfusaoMedia);
        simple = tsmovavg(PerfusaoMedia','s',mediamovel,1);
        hold on;
        plot(simple);
        grid on; grid minor;
        xlabel('# Frames');
        ylabel('$$P_b = {(1-K)\over K}$$','Interpreter', 'Latex');
        
        %% Gráfico Perfusao Normalizada:
        
        Norm_Perfusao=(PerfusaoMedia-min(PerfusaoMedia))/(max(PerfusaoMedia)-min(PerfusaoMedia));
        figure; 
        plot (Norm_Perfusao);
        simple = tsmovavg(Norm_Perfusao','s',mediamovel,1);
        hold on;
        plot(simple);
        grid on; grid minor;
        xlabel('# Frames');
      
end
            
        
        
                    
end
