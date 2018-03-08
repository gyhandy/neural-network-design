%author Geyunhao

clc;
clear all;
%��ȡ����
data=load('hw4-data.txt');
[data_row,data_clown]=size(data);

%�����֯ӳ������m*n
m=5;
n=5;
%��Ԫ�ڵ�����som_sum
neuron_sum=m*n;
%Ȩֵ��ʼ���������ʼ��
w = rand(neuron_sum, data_clown);
%��ʼ��ѧϰ��
learn_init = 0.5;
learn_rate = learn_init;
%ѧϰ�ʲ���
learn_index=1000;
%���õ�������
iter =500;
%��Ԫλ������ת����arroy to matrix��(clown priority)
[I,J] = ind2sub([m, n], 1:neuron_sum);
%�����ʼ�� 
neighbor_init =2;
neighbor_redius = neighbor_init;
%�������
neighbor_index = 1000/log(neighbor_init);

%��������
for t=1:iter 
    %  ���������
    for j=1:data_row  
        %��ȡ������ֵ
        data_x = data(j,:); 
        %�ҵ���ʤ��Ԫ
        [win_row, win_som_index]=min(dist(data_x,w'));  
        %��ʤ��Ԫ������λ��
        [win_som_row,win_som_cloumn] =  ind2sub([m, n],win_som_index);
        win_som=[win_som_row,win_som_cloumn];
        %����������Ԫ�ͻ�ʤ��Ԫ�ľ���,������
        %distance_som = sum(( ([I( : ), J( : )] - repmat(win_som, som_sum,1)) .^2) ,2);
        %guass function
        distance_som = exp( sum(( ([I( : ), J( : )] - repmat(win_som, neuron_sum,1)) .^2) ,2)/(-2*neighbor_redius*neighbor_redius)) ;
        %Ȩֵ����
        for i = 1:neuron_sum
          % if distance_som(i)<neighbor_redius*neighbor_redius 
            w(i,:) = w(i,:) + learn_rate.*distance_som(i).*( data_x - w(i,:));
        end
    end

    %����ѧϰ��
    learn_rate = learn_init * exp(-t/learn_index);   
    %��������뾶
    neighbor_redius = neighbor_init*exp(-t/neighbor_index);  
end
%data��������Ԫ��ӳ��
%��Ԫ����som_num�洢ͼ����
som_num=cell(1,size(w,1));
for i=1:size(w,1)
    som_num{1,i}=[];
end
%ÿ����Ԫ�ڵ��Ӧ��data�������
for num=1:data_row
%     [som_row,clown]= min(sum(( (w - repmat(data(num,:), neuron_sum,1)) .^2) ,2));
%     som_num{1,clown}= [som_num{1,clown},num];    
    [value,position]= min(sum(( (w - repmat(data(num,:), neuron_sum,1)) .^2) ,2));
    %rowΪ��Ӧ����Ԫ
    som_num{1,position}= [som_num{1,position},num]; %ֱ���ں������ 
end

%�洢��Ԫ���飬.txt��ʽ
%save_som_data(file_path,som_sum,som_num);
%�洢��Ԫ���飬.mat��ʽ
% path1=strcat('som.mat');
save('som.mat','som_num');
