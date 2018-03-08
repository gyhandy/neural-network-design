%author Yunhao Ge
%student number 116020910017
%%生成数据
clear;
clc;
twospiraltrain=load('two_spiral_train.txt');
twospiraltest=load('two_spiral_test.txt');
train_number=size(twospiraltrain,1);%训练数据个数
train_circle_number=5000;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
test_number=size(twospiraltrain,1);%测试数据个数
train_label1=twospiraltrain(:,end)';   %训练数据类别，
train_data1=twospiraltrain(:,1:end-1)'; %训练数据属性，


posiNum=find(train_label1(:)==1);
positrain_data = train_data1(:,posiNum);%找到正螺旋
nagNum=find(train_label1(:)==0);
nagtrain_data = train_data1(:,nagNum);%找到反螺旋
        
for i=1:train_number
    switch train_label1(i)
        case 0
            train_label2(i,:)=[1 0];
        case 1
            train_label2(i,:)=[0 1];
    end
end
train_label=train_label2'; %将每个标签表示为1*2的矩阵，输出由两个神经元来表示。        
plot(positrain_data(1,:),positrain_data(2,:),'r+');%画出图像
hold on;
plot(nagtrain_data(1,:),nagtrain_data(2,:),'go');
 %% BP神经网络结构的初始化
  %网络结构：2个输入神经元，10个隐含层神经元，2个输出神经元
  innum=2;
  hidnum=10;
  outnum=2;
  [train_data,train_datas]=mapminmax(train_data1);%将数据作归一化处理
  %输入输出取值阈值随机初始化
  %w1矩阵表示每一行为一个隐含层神经元的输入权值
  w1=rands(hidnum,innum);%rands函数用来初始化神经元的权值和阈值是很合适的,w
  b1=rands(hidnum,1);%b1为10*1的矩阵，一个神经元对应一行
  %w2矩阵表示每一列为一个输出层神经元的输入权值
  w2=rands(hidnum,outnum);%w2为10*2的矩阵
  b2=rands(outnum,1);%b2为2*1的矩阵
 
  %用来保存上一次的权值和阈值，w1和w2均为运算过程中的，迭代过的数据滚动保留 两次，因为后面的更新方差是递归的
  w1_1=w1;w1_2=w1_1;
  b1_1=b1;b1_2=b1_1;
  w2_1=w2;w2_2=w2_1;
  b2_1=b2;b2_2=b2_1;
  
  %学习率的设定
  alpha=0.1;
  
  %计划训练5000次终止训练。
  for train_circle=1:train_circle_number
      for i=1:train_number%训练样本
         %% 输入层的输出
          x=train_data(:,i);%取出第i个样本，x(i)为2*1的列向量
         %% 隐含层的输出 
          I=w1*x+b1;%隐含层的输出，输出层的输入
          Iout=(1./(1+exp(-I)))'; %采用logsig函数
         %% 输出层的输出
          yn=(Iout*w2)'+b2;   %yn为2*1的列向量，因此此时的传函为线性的，所以可以一步到位，不必上面
          
         %% 计算误差
          e=train_label(:,i)-yn; %e为2*1的列向量，保存的是误差值
         %% 计算权值变换率
          %输出层的权值变化率就是误差评价函数对权值的导数课本P202
          dw2=e*Iout; %dw2为2*10的矩阵，每一行表示输出接点的输入权值变化率
          db2=e'; %e为1*2的行向量
          
          for j=1:hidnum
              S=1/(1+exp(-I(j)));%经过函数以后的
             FI(j)=S*(1-S);  %FI(j)为一实数，FI为1*10的行向量%每一层输出函数的输入变量的偏导
         end
         
         for k=1:1:innum
             for j=1:hidnum
                  dw1(k,j)=FI(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2));    %dw1为2*6的矩阵
                  db1(j)=FI(j)*(e(1)*w2(j,1)+e(2)*w2(j,2));   %db1为1*6的矩阵
              end
          end
         
         %% 权值更新方程
         w1=w1_1+alpha*dw1'; %w1仍为10*2的矩阵
         b1=b1_1+alpha*db1'; %b1仍为10*1的矩阵
         w2=w2_1+alpha*dw2'; %w2仍为10*2的矩阵
         b2=b2_1+alpha*db2'; %b2仍为10*1的矩阵
         
         %% 保存上一次的权值和阈值
         w1_2=w1_1;w1_1=w1;
         b1_2=b1_1;b1_1=b1;
         w2_2=w2_1;w2_1=w2;
         b2_2=b2_1;b2_1=b2;
     end
 end 
 %%读取训练集
test_number=size(twospiraltest,1);%训练数据个数
test_circle_number=1000;%训练过程学习次数
test_number=size(twospiraltest,1);%测试数据个数
test_label1=twospiraltest(:,end)';   %训练数据类别，
test_data1=twospiraltest(:,1:end-1)'; %训练数据属性，

testposiNum=find(test_label1(:)==1);
positest_data = test_data1(:,testposiNum);%找到正螺旋
testnagNum=find(test_label1(:)==0);
nagtest_data = test_data1(:,testnagNum);%找到反螺旋
        
for i=1:test_number
    switch test_label1(i)
        case 0
            test_label2(i,:)=[1 0];
        case 1
            test_label2(i,:)=[0 1];
    end
end
test_label=test_label2'; %将每个标签表示为1*2的矩阵，输出由两个神经元来表示。     
plot(positest_data(1,:),positest_data(2,:),'c+');
hold on;
plot(nagtest_data(1,:),nagtest_data(2,:),'bo');



 legend('螺旋线1训练','螺旋线2训练','螺旋线1训练','螺旋线2训练')
 test_data=mapminmax('apply',test_data1,train_datas);
 
 %% 用训练到的模型预测数据
 for i=1:test_number
%       I=w1*x+b1;%隐含层的输出，输出层的输入
%       Iout=(1./(1+exp(-I)))'; %采用logsig函数
     for j=1:hidnum
         I(j)=test_data(:,i)'*w1(j,:)'+b1(j);
         Iout(j)=1/(1+exp(-I(j)));%Iout为1*10的行向量
     end
     predict(:,i)=w2'*Iout'+b2;%predict
 end
 %% 预测结果分析
 for i=1:test_number
     output_pred(i)=find(predict(:,i)==max(predict(:,i)));    
 end
 error=output_pred-test_label1-1;    %
 %% 计算出每一类预测错误的个数总和
 k=zeros(1,2); %k=[0 0]
 for i=1:test_number
     if error(i)~=0    %matlab中不能用if error(i)！=0 
         [b c]=max(test_label(:,i));
         switch c
             case 1
                 k(1)=k(1)+1;
             case 2
                 k(2)=k(2)+1;
         end
     end
 end
 %% 求出每一类总体的个数和
 kk=zeros(1,2); %k=[0 0]
 for i=1:test_number
     [b c]=max(test_label(:,i));
     switch c
         case 1
             kk(1)=kk(1)+1;
         case 2
             kk(2)=kk(2)+1;
     end
end
%% 计算每一类的正确率
 accuracy_rate=(kk-k)./kk