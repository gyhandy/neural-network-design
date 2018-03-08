% one versus rest
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
   
%# train one-against-all models   
model = cell(numLabels,1); %模型的个数   
for k=1:numLabels   
    model{k} = libsvmtrain(double(trainLabel==(k-1)), trainData, '-c 1 -g 0.2 -t 0 -b 1');   
end   
%# get probability estimates of test instances using each model   
prob = zeros(numTest,numLabels);   
for k=1:numLabels   
    [~,~,p] = libsvmpredict(double(testLabel==(k-1)), testData, model{k}, '-b 1');   
    prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k   
end   
   
%# predict the class with the highest probability   
[~,pred] = max(prob,[],2);%取出预测最大值对应的标签
pred1=pred-1;
acc =sum(pred1 == testLabel) ./ numel(testLabel)    %# accuracy   
% C = confusionmat(testLabel, pred)                   %# confusion matrix  