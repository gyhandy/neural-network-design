%author Yunhao Ge
%student number 116020910017
%%��������
clear;
clc;
twospiraltrain=load('two_spiral_train.txt');
twospiraltest=load('two_spiral_test.txt');
train_number=size(twospiraltrain,1);%ѵ�����ݸ���
train_circle_number=5000;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
test_number=size(twospiraltrain,1);%�������ݸ���
train_label1=twospiraltrain(:,end)';   %ѵ���������
train_data1=twospiraltrain(:,1:end-1)'; %ѵ���������ԣ�


posiNum=find(train_label1(:)==1);
positrain_data = train_data1(:,posiNum);%�ҵ�������
nagNum=find(train_label1(:)==0);
nagtrain_data = train_data1(:,nagNum);%�ҵ�������
        
for i=1:train_number
    switch train_label1(i)
        case 0
            train_label2(i,:)=[1 0];
        case 1
            train_label2(i,:)=[0 1];
    end
end
train_label=train_label2'; %��ÿ����ǩ��ʾΪ1*2�ľ��������������Ԫ����ʾ��        
plot(positrain_data(1,:),positrain_data(2,:),'r+');%����ͼ��
hold on;
plot(nagtrain_data(1,:),nagtrain_data(2,:),'go');
 %% BP������ṹ�ĳ�ʼ��
  %����ṹ��2��������Ԫ��10����������Ԫ��2�������Ԫ
  innum=2;
  hidnum=10;
  outnum=2;
  [train_data,train_datas]=mapminmax(train_data1);%����������һ������
  %�������ȡֵ��ֵ�����ʼ��
  %w1�����ʾÿһ��Ϊһ����������Ԫ������Ȩֵ
  w1=rands(hidnum,innum);%rands����������ʼ����Ԫ��Ȩֵ����ֵ�Ǻܺ��ʵ�,w
  b1=rands(hidnum,1);%b1Ϊ10*1�ľ���һ����Ԫ��Ӧһ��
  %w2�����ʾÿһ��Ϊһ���������Ԫ������Ȩֵ
  w2=rands(hidnum,outnum);%w2Ϊ10*2�ľ���
  b2=rands(outnum,1);%b2Ϊ2*1�ľ���
 
  %����������һ�ε�Ȩֵ����ֵ��w1��w2��Ϊ��������еģ������������ݹ������� ���Σ���Ϊ����ĸ��·����ǵݹ��
  w1_1=w1;w1_2=w1_1;
  b1_1=b1;b1_2=b1_1;
  w2_1=w2;w2_2=w2_1;
  b2_1=b2;b2_2=b2_1;
  
  %ѧϰ�ʵ��趨
  alpha=0.1;
  
  %�ƻ�ѵ��5000����ֹѵ����
  for train_circle=1:train_circle_number
      for i=1:train_number%ѵ������
         %% ���������
          x=train_data(:,i);%ȡ����i��������x(i)Ϊ2*1��������
         %% ���������� 
          I=w1*x+b1;%����������������������
          Iout=(1./(1+exp(-I)))'; %����logsig����
         %% ���������
          yn=(Iout*w2)'+b2;   %ynΪ2*1������������˴�ʱ�Ĵ���Ϊ���Եģ����Կ���һ����λ����������
          
         %% �������
          e=train_label(:,i)-yn; %eΪ2*1��������������������ֵ
         %% ����Ȩֵ�任��
          %������Ȩֵ�仯�ʾ���������ۺ�����Ȩֵ�ĵ����α�P202
          dw2=e*Iout; %dw2Ϊ2*10�ľ���ÿһ�б�ʾ����ӵ������Ȩֵ�仯��
          db2=e'; %eΪ1*2��������
          
          for j=1:hidnum
              S=1/(1+exp(-I(j)));%���������Ժ��
             FI(j)=S*(1-S);  %FI(j)Ϊһʵ����FIΪ1*10��������%ÿһ��������������������ƫ��
         end
         
         for k=1:1:innum
             for j=1:hidnum
                  dw1(k,j)=FI(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2));    %dw1Ϊ2*6�ľ���
                  db1(j)=FI(j)*(e(1)*w2(j,1)+e(2)*w2(j,2));   %db1Ϊ1*6�ľ���
              end
          end
         
         %% Ȩֵ���·���
         w1=w1_1+alpha*dw1'; %w1��Ϊ10*2�ľ���
         b1=b1_1+alpha*db1'; %b1��Ϊ10*1�ľ���
         w2=w2_1+alpha*dw2'; %w2��Ϊ10*2�ľ���
         b2=b2_1+alpha*db2'; %b2��Ϊ10*1�ľ���
         
         %% ������һ�ε�Ȩֵ����ֵ
         w1_2=w1_1;w1_1=w1;
         b1_2=b1_1;b1_1=b1;
         w2_2=w2_1;w2_1=w2;
         b2_2=b2_1;b2_1=b2;
     end
 end 
 %%��ȡѵ����
test_number=size(twospiraltest,1);%ѵ�����ݸ���
test_circle_number=1000;%ѵ������ѧϰ����
test_number=size(twospiraltest,1);%�������ݸ���
test_label1=twospiraltest(:,end)';   %ѵ���������
test_data1=twospiraltest(:,1:end-1)'; %ѵ���������ԣ�

testposiNum=find(test_label1(:)==1);
positest_data = test_data1(:,testposiNum);%�ҵ�������
testnagNum=find(test_label1(:)==0);
nagtest_data = test_data1(:,testnagNum);%�ҵ�������
        
for i=1:test_number
    switch test_label1(i)
        case 0
            test_label2(i,:)=[1 0];
        case 1
            test_label2(i,:)=[0 1];
    end
end
test_label=test_label2'; %��ÿ����ǩ��ʾΪ1*2�ľ��������������Ԫ����ʾ��     
plot(positest_data(1,:),positest_data(2,:),'c+');
hold on;
plot(nagtest_data(1,:),nagtest_data(2,:),'bo');



 legend('������1ѵ��','������2ѵ��','������1ѵ��','������2ѵ��')
 test_data=mapminmax('apply',test_data1,train_datas);
 
 %% ��ѵ������ģ��Ԥ������
 for i=1:test_number
%       I=w1*x+b1;%����������������������
%       Iout=(1./(1+exp(-I)))'; %����logsig����
     for j=1:hidnum
         I(j)=test_data(:,i)'*w1(j,:)'+b1(j);
         Iout(j)=1/(1+exp(-I(j)));%IoutΪ1*10��������
     end
     predict(:,i)=w2'*Iout'+b2;%predict
 end
 %% Ԥ��������
 for i=1:test_number
     output_pred(i)=find(predict(:,i)==max(predict(:,i)));    
 end
 error=output_pred-test_label1-1;    %
 %% �����ÿһ��Ԥ�����ĸ����ܺ�
 k=zeros(1,2); %k=[0 0]
 for i=1:test_number
     if error(i)~=0    %matlab�в�����if error(i)��=0 
         [b c]=max(test_label(:,i));
         switch c
             case 1
                 k(1)=k(1)+1;
             case 2
                 k(2)=k(2)+1;
         end
     end
 end
 %% ���ÿһ������ĸ�����
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
%% ����ÿһ�����ȷ��
 accuracy_rate=(kk-k)./kk