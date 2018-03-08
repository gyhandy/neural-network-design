clc;
clear all;
%读取数据并作分类
twospiraltrain=load('two_spiral_train.txt');
train_number=size(twospiraltrain,1);%训练数据个数                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
train_label1=twospiraltrain(:,end)';   %训练数据类别
train_data1=twospiraltrain(:,1:end-1)'; %训练数据

%训练模型
index = MLQP_function(train_data1,train_label1);

% posiNum=find(train_label1(:)==1);
% positrain_data = train_data1(:,posiNum);%找到正螺旋
% nagNum=find(train_label1(:)==0);
% nagtrain_data = train_data1(:,nagNum);%找到反螺旋

% %%分组a b 均为1，c d 均为0
% % 正螺旋ab（1）
% posiSum=size(positrain_data,2);
% rand_posi=randperm(posiSum,posiSum);
% a_num = rand_posi(:,1:size(rand_posi,2)/2);%第一组a
% a = positrain_data(:,a_num)';
% a_lable = ones(size(a,1),1);
% b_num = rand_posi(:,size(rand_posi,2)/2+1:size(rand_posi,2));%第二组b
% b = positrain_data(:,b_num)';
% b_lable = ones(size(b,1),1);
% % 反螺旋cd（2）
% nagSum=size(nagtrain_data,2);
% rand_nag=randperm(nagSum,nagSum);
% c_num = rand_nag(:,1:size(rand_nag,2)/2);%第一组a
% c = nagtrain_data(:,c_num)';
% c_lable = zeros(size(c,1),1);
% d_num = rand_nag(:,size(rand_nag,2)/2+1:size(rand_nag,2));%第二组b
% d = nagtrain_data(:,d_num)';
% d_lable = zeros(size(d,1),1);
%作图
% figure;
% for i=1:1:size(a,1)
%     hold on;
%     plot(a(i,1),a(i,2),'ro');
%     axis([-4 4 -4 4]);
% end
% figure;
% for i=1:1:size(b,1)
%     hold on;
%     plot(b(i,1),b(i,2),'r+');
%     axis([-4 4 -4 4]);
% end
% figure;
% for i=1:1:size(c,1)
%      hold on;
%      plot(c(i,1),c(i,2),'go');
%      axis([-4 4 -4 4]);
% end
% figure;
% for i=1:1:size(a,1)
%      hold on;
%      plot(d(i,1),d(i,2),'g+');
%      axis([-4 4 -4 4]);
% end
% %%
% %分别学习[ac][ad][bc][bd]
% %[ac]
% ac=[a;c]';
% ac_lable=[a_lable;c_lable]';
% index_ac= MLQP_function(ac,ac_lable);
% %[ad]
% ad=[a;d]';
% ad_lable=[a_lable;d_lable]';
% index_ad= MLQP_function(ad,ad_lable);
% %[bc]
% bc=[b;c]';
% bc_lable=[b_lable;c_lable]';
% index_bc= MLQP_function(bc,bc_lable);
% %[bd]
% bd=[b;d]';
% bd_lable=[b_lable;d_lable]';
% index_bd= MLQP_function(bd,bd_lable);

%%
% 读取测试集
twospiraltest=load('two_spiral_test.txt');
test_number=size(twospiraltrain,1);%测试数据个数
test_label1=twospiraltest(:,end);   %测试数据标签
test_data1=twospiraltest(:,1:end-1); %测试数据
predict=zeros(test_number,size(test_label1,2));

for j=1:test_number %遍历所有样本
     test_data=test_data1(j,:);%读取一个测试样本
     test_lable=test_label1(j,:);%读取一个测试标签
     
     predict(j,:)=testing(index,test_data);
end


 figure;
for z=1:1:test_number
    if predict(z,:)==1;
        hold on;
        plot(test_data1(z,1),test_data1(z,2),'ro');
    else
        hold on;
        plot(test_data1(z,1),test_data1(z,2),'bo');
    end
end
%  
% 
% 
% 
% 
