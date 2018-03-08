% one versus one
clc;
clear all;
%%读取训练集   
[origtrain_label,origtrain_data] = libsvmread('train.txt');%读取数据到matlab格式   
numTrain = size(origtrain_data,1);   
numLabels = max(origtrain_label)+1;    
% split training 打乱训练顺序  
idx_train = randperm(numTrain);  
trainData = origtrain_data(idx_train(1:numTrain),:);    
trainLabel = origtrain_label(idx_train(1:numTrain)); 

%分成12组
mindata = cell(numLabels,1);
minlable = cell(numLabels,1);
for k=1:numLabels
    j=1;
    for i=1:numTrain
        if origtrain_label(i) == (k-1)
           mindata{k}(j,:) = origtrain_data(i,:);
           minlable{k}(j,:) = k-1;
           j=j+1;
        end
    end
end

%one versue one 训练
model=cell(numLabels*(numLabels-1)/2,1);
x=1;
for m=1:numLabels
    for n=1:numLabels
        if m<n
           onedata=[mindata{m};mindata{n}];
           onelable=[minlable{m};minlable{n}];
           model{x} = libsvmtrain(onelable, onedata, '-c 1 -g 0.2 -t 0 -b 1'); 
           x=x+1;
        end
        
    end
end

%%读取测试集   
[origtest_label,origtest_data] = libsvmread('test.txt');%读取数据到matlab格式   
numTest = size(origtest_data,1);    
%# split training 打乱测试顺序  
idx_test = randperm(numTest);  
testData = origtest_data(idx_test(1:numTest),:);    
testLabel = origtest_label(idx_test(1:numTest));  

%进行预测
predict_label = zeros(numTest,x-1);   
for i=1:x-1   
    [predict_label(:,i), accuracy, prob] = libsvmpredict(testLabel, testData, model{i}, '-b 1');   
    
end   
predict=mode(predict_label,2);
acc =sum(predict == testLabel) ./ numel(testLabel)    %# accuracy   
