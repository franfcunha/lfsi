cd 'C:\Users\utilizador\Desktop\LFToolbox0.4\'

LFMatlabPathSetup();
%LFUtilProcessWhiteImages();

cd 'C:\Users\utilizador\Desktop\Samples'
LFUtilDecodeLytroFolder('Cameras\B5143107340\CalSample\');


CalOptions.ExpectedCheckerSize = [15, 10];
CalOptions.ExpectedCheckerSpacing_m = 1e-3*[25, 25];

LFUtilCalLensletCam('Cameras/B5143107340/CalSample',CalOptions);

