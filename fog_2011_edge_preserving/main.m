#!/usr/bin/octave
pkg load image

RGB = imread('v.jpg'); 
I = double(RGB);
I = I./max(I(:));
res = wlsFilter(I, 0.5);
%figure, imshow(I), figure, imshow(res)
%res = wlsFilter(I, 2, 2);
imwrite(res,'v2.jpg')
%figure, imshow(res)
fprintf('Done')