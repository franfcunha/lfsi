



function [contrast] = lasca(imagem,n)

% img is the 2-D speckle image to be filtered
kernel = ones(n,n);% These dimensions are arbitrary
Nk = sum(kernel(:));
mu_img = filter2(kernel,imagem,'same')/Nk;
img_sq = filter2(kernel,imagem.^2,'same');
sig_img = sqrt( (img_sq - Nk*mu_img.^2)/(Nk-1) );
contrast = sig_img./mu_img;% local contrast

end