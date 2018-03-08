% one versus one
clc;
clear all;
%%读取训练集   
[origtrain_label,origtrain_data] = libsvmread('train.txt');%读取数据到matlab格式   
numTrain = size(origtrain_data,1);   
numLabels = max(origtrain_label)+1;    
%# split training 打乱训练顺序  
idx_train = randperm(numTrain);  
trainData = origtrain_data(idx_train(1:numTrain),:);    
trainLabel = origtrain_label(idx_train(1:numTrain));   

%%读取测试集   
[origtest_label,origtest_data] = libsvmread('test.txt');%读取数据到matlab格式   
numTest = size(origtest_data,1);    
%# split training 打乱测试顺序  
idx_test = randperm(numTest);  
testData = origtest_data(idx_test(1:numTest),:);    
testLabel = origtest_label(idx_test(1:numTest));   
   
%# train one-against-one models   
model = libsvmtrain(trainLabel, trainData, '-c 1 -g 0.2 -t 0 -b 1');   
  
%# get probability estimates of test instances using each model   
[predict_label, accuracy, prob] = libsvmpredict(testLabel, testData, model, '-b 1');   
  
   
% clc;   
% clear all;   
%    
% [iris_label,iris_data] = libsvmread('iris.scale');%读取数据到matlab格式   
% % [~,~,labels] = unique(species);   %# labels: 1/2/3   
% % data = zscore(meas);              %# scale features   
% numInst = size(iris_data,1);   
% numLabels = max(iris_label);   
%    
% %# split training/testing   
% idx = randperm(numInst);   
% numTrain = 100;   
% numTest = numInst - numTrain;   
% trainData = iris_data(idx(1:numTrain),:);    
% testData = iris_data(idx(numTrain+1:end),:);   
% trainLabel = iris_label(idx(1:numTrain));   
% testLabel = iris_label(idx(numTrain+1:end));   
%    
% model= svmtrain(trainLabel, trainData, '-c 1 -g 0.2 -b 1');   
% [predict_label, accuracy, prob] = svmpredict(testLabel,testData, model,'-b 1');   
% % fprintf('准确率为%d.....\n',accuracy);  