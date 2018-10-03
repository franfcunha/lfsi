
%{ 
Example of execution; the resulting stack of frames will be stored in
'frames' variable after execution of the function with the following syntax 

frames=copula_simulation(L,D,#frames,type);

L-size of the squared matrix; (base)
D-diameter of the aperture; (diameter)
#frames - number of frames; (N_frames)
type- 's' for subjective speckle simulation; 'o' for objective speckle
simulation (tipo)
% 
Example:

frames = copula_simulation(256,128,50,'s');

%}

function [speckle_cube] = copula_simulation(base,diameter,N_frames,tipo)
    
    %initialization
    
    speckle_cube = zeros(base,base,N_frames);
    Z = zeros(base,base,N_frames);
    
    % imaging pupil used for subjective speckle generation
   
    [x,y]=meshgrid(-base/2 : base/2-1,-base/2 : base/2-1);  
    z=sqrt(x.^2 + y.^2);
    c= z<(diameter/2);
    
    % X1 and X2 generation
    
    rng('shuffle')
    X1=rand(base);
    rng('shuffle')
    X2=rand(base);
    
    %start the temporal loop
    i=1;
    
    while i<=N_frames
        
        % equation 17 of the paper
        
        Z(:,:,i)=sqrt(-2*log(X1)).*cos(2*pi*X2 +((pi/2)*((i-1)/(N_frames-1))));
        
        %percentile transformation on each resulting Z matrix
               
        T=normcdf(Z(:,:,i),mean2(Z(:,:,i)));
        
        % assign amplitude and phase values accordingly to Euler's formula
        
        phasor=complex(cos(2*pi*12*T),-sin(2*pi*12*T)); 
        
     
        % Fourier Transformation
        
        TF_final = fftshift(fft2(phasor)); 
        
        speckle_cube(:,:,i)=TF_final;
        
        i=i+1;
       
    end
    
    %objective speckle generation 
    if tipo=='o'
    
       for j=1:size(speckle_cube,3)
           intensidade=log(1+abs(speckle_cube(:,:,j)));
           i_max=max(intensidade(:));
           speckle_cube(:,:,j)=uint8(255 * (intensidade/i_max));
       end
   % subjective speckle generation    
   elseif tipo =='s'
        for j=1:size(speckle_cube,3)
            % convolution corresponds to the multiplication of the pupil
            % imaging function, c, in the fourier domain 
            intensidade=abs(ifft2(speckle_cube(:,:,j).*c)).^2;
            i_max=max(intensidade(:));
            speckle_cube(:,:,j)=uint8(255 * (intensidade/i_max));
        end
   end
end
