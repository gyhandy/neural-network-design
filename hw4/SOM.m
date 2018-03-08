%author Geyunhao

clc;
clear all;
%读取数据
data=load('hw4-data.txt');
[data_row,data_clown]=size(data);

%搭建自组织映射网络m*n
m=5;
n=5;
%神经元节点总数som_sum
neuron_sum=m*n;
%权值初始化，随机初始化
w = rand(neuron_sum, data_clown);
%初始化学习率
learn_init = 0.5;
learn_rate = learn_init;
%学习率参数
learn_index=1000;
%设置迭代次数
iter =500;
%神经元位置坐标转换（arroy to matrix）(clown priority)
[I,J] = ind2sub([m, n], 1:neuron_sum);
%邻域初始化 
neighbor_init =2;
neighbor_redius = neighbor_init;
%邻域参数
neighbor_index = 1000/log(neighbor_init);

%迭代次数
for t=1:iter 
    %  样本点遍历
    for j=1:data_row  
        %获取样本点值
        data_x = data(j,:); 
        %找到获胜神经元
        [win_row, win_som_index]=min(dist(data_x,w'));  
        %获胜神经元的拓扑位置
        [win_som_row,win_som_cloumn] =  ind2sub([m, n],win_som_index);
        win_som=[win_som_row,win_som_cloumn];
        %计算其他神经元和获胜神经元的距离,邻域函数
        %distance_som = sum(( ([I( : ), J( : )] - repmat(win_som, som_sum,1)) .^2) ,2);
        %guass function
        distance_som = exp( sum(( ([I( : ), J( : )] - repmat(win_som, neuron_sum,1)) .^2) ,2)/(-2*neighbor_redius*neighbor_redius)) ;
        %权值更新
        for i = 1:neuron_sum
          % if distance_som(i)<neighbor_redius*neighbor_redius 
            w(i,:) = w(i,:) + learn_rate.*distance_som(i).*( data_x - w(i,:));
        end
    end

    %更新学习率
    learn_rate = learn_init * exp(-t/learn_index);   
    %更新邻域半径
    neighbor_redius = neighbor_init*exp(-t/neighbor_index);  
end
%data数据在神经元的映射
%神经元数组som_num存储图像编号
som_num=cell(1,size(w,1));
for i=1:size(w,1)
    som_num{1,i}=[];
end
%每个神经元节点对应的data样本编号
for num=1:data_row
%     [som_row,clown]= min(sum(( (w - repmat(data(num,:), neuron_sum,1)) .^2) ,2));
%     som_num{1,clown}= [som_num{1,clown},num];    
    [value,position]= min(sum(( (w - repmat(data(num,:), neuron_sum,1)) .^2) ,2));
    %row为对应的神经元
    som_num{1,position}= [som_num{1,position},num]; %直接在后面添加 
end

%存储神经元数组，.txt格式
%save_som_data(file_path,som_sum,som_num);
%存储神经元数组，.mat格式
% path1=strcat('som.mat');
save('som.mat','som_num');
