%make pca for the w of nueron
clc;
clear all;
load('w.mat');
retain_dimensions = 2;
[U,S,V] = svd(cov(w));
reduced_w = w*U(:,1:retain_dimensions);
% [reduced_nor_w,PS_w]=mapminmax(reduced_w);

for i=1:size(reduced_w,1)
    plot(reduced_w(i,1),reduced_w(i,2),'ro');
    hold on;
end
%make pca for the EEG data
load('hw4-EEG.mat');
retain_dimensions = 2;
[U,S,V] = svd(cov(EEG_X));
reduced_EEG_X = EEG_X*U(:,1:retain_dimensions);
figure;
for i=1:size(reduced_EEG_X,1);
    plot(reduced_EEG_X(i,1),reduced_EEG_X(i,2),'bo');
    hold on;
end