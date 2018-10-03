
function [frames] = reduz_contraste(frames_simulados, inicio_reducao, fim_reducao,sigma_i, sigma_f)
    
    n_frames_alterados=fim_reducao - inicio_reducao;
    n_frames_recov = floor(1.25 * n_frames_alterados);
    
    delta_sigma= sigma_f - sigma_i;
    
    sigma = sigma_i + delta_sigma/n_frames_alterados * linspace(1,n_frames_alterados, n_frames_alterados);
    sigma_recov= interp1(linspace(0,1,numel(sigma)),fliplr(sigma),linspace(0,1,n_frames_recov));
    sigma=horzcat(sigma,sigma_recov);
    
    frames=frames_simulados;
    L=size(frames,1);
    
    a=.9;
    b=1;
    
    for i=inicio_reducao:(inicio_reducao+n_frames_alterados+n_frames_recov-1)
       disp(i);
       rng('shuffle')
       frames(:,:,i)= imgaussfilt(frames(:,:,i), sigma(i-(inicio_reducao-1))).*((b-a).*rand(L) + a);
        
    end

end



% nova_simulacao=reduz_contraste(copula_simulado,frame_inicio_perfusao,frame_pico_perfusao,sigma_i,sigma_f)

% eu usei sigma de 0.25 a 1.25