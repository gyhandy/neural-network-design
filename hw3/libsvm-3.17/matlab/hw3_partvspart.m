% part versus part
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
model=cell(numLabels*(numLabels-1)/2,4);
x=1;
for m=1:numLabels
    for n=1:numLabels
        if m<n
%            onedata=[mindata{m};mindata{n}];
%            onelable=[minlable{m};minlable{n}];
           %mn分为part
           a_datanum = round(size(mindata{m},1)/2);
           a_data = mindata{m}(1:a_datanum,:);
           a_lable = minlable{m}(1:a_datanum,:);
           b_data = mindata{m}(a_datanum+1:end,:);
           b_lable = minlable{m}(a_datanum+1:end,:);
           
           c_datanum = round(size(mindata{n},1)/2);
           c_data = mindata{n}(1:c_datanum,:);
           c_lable = minlable{n}(1:c_datanum,:);
           d_data = mindata{n}(c_datanum+1:end,:);
           d_lable = minlable{n}(c_datanum+1:end,:);

           %ac=1,ad=2,bc=3,bd=4
           ac_partdata=[a_data;c_data];
           ac_partlable=[a_lable;c_lable];
           model{x,1} = libsvmtrain(ac_partlable, ac_partdata, '-c 1 -g 0.2 -t 2 -b 1');
           
           ad_partdata=[a_data;d_data];
           ad_partlable=[a_lable;d_lable];
           model{x,2} = libsvmtrain(ad_partlable, ad_partdata, '-c 1 -g 0.2 -t 2 -b 1');
           
           bc_partdata=[b_data;c_data];
           bc_partlable=[b_lable;c_lable];
           model{x,3} = libsvmtrain(bc_partlable, bc_partdata, '-c 1 -g 0.2 -t 2 -b 1');
          
           bd_partdata=[b_data;d_data];
           bd_partlable=[b_lable;d_lable];
           model{x,4} = libsvmtrain(bd_partlable, bd_partdata, '-c 1 -g 0.2 -t 2 -b 1');
           
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

%part vs part进行预测
predict_label = zeros(numTest,x-1);
predict_mid = zeros(numTest,4);
predict_min = zeros(numTest,2);
for i=1:x-1   
    
    [predict_mid(:,1), accuracy, prob] = libsvmpredict(testLabel, testData, model{i,1}, '-b 1');
    [predict_mid(:,2), accuracy, prob] = libsvmpredict(testLabel, testData, model{i,2}, '-b 1');
    [predict_mid(:,3), accuracy, prob] = libsvmpredict(testLabel, testData, model{i,3}, '-b 1');
    [predict_mid(:,4), accuracy, prob] = libsvmpredict(testLabel, testData, model{i,4}, '-b 1');
    
    %min(model{i,1}.Label(1))[ac,ad]
        
    
        for e=1:size(testData,1)
        if predict_mid(e,1)==model{i,1}.Label(1) && predict_mid(e,2)==model{i,1}.Label(1);
           predict_min(e,1)=model{i,1}.Label(1);
        else
           predict_min(e,1)=model{i,1}.Label(2);
        end
        end
       
    %min(model{i,1}.Label(1))[bc,bd]
        for e=1:size(testData,1)
        if predict_mid(e,3)==model{i,1}.Label(1) && predict_mid(e,4)==model{i,1}.Label(1);
           predict_min(e,2)=model{i,1}.Label(1);
        else
           predict_min(e,2)=model{i,1}.Label(2);
        end
        end
        
         %max(model{i,1}.Label(2)) abcd
         for e=1:size(testData,1)
         if predict_min(e,1)==model{i,1}.Label(2) && predict_min(e,2)==model{i,1}.Label(2);;
           predict_label(e,i)=model{i,1}.Label(2);
         else
           predict_label(e,i)=model{i,1}.Label(1);
         end
         end
    
    
    [predict_label(:,i), accuracy, prob] = libsvmpredict(testLabel, testData, model{i}, '-b 1');    
end   
 predict=mode(predict_label,2);
acc =sum(predict == testLabel) ./ numel(testLabel)    %# accuracy   
