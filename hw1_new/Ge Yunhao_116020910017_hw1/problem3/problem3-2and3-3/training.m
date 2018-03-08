%前向传递网络，得到输出误差e，与评价函数F
%全部用矩阵运算
function [index] =training(index,net_configure,train_data,train_lable,learningrate)%部获取
net_layer=(size(net_configure,2)-1);
u=1;
v=2;
b=3;
x1=train_data;%取一个输入
t=train_lable;%取一个标签
z=cell(1,net_layer);%z每一层神经元都有，而且数目不一定一样，所以用cell
x=cell(1,net_layer);
for i=1:net_layer%遍历每一层神经元
    z1=(index{u,i}*x1.^2)+(index{v,i}*x1)+index{b,i};%z全部保存，修正的时候用
    x1=sigmoid(z1); %最后保留的数据
    z{1,i}=z1;
    x{1,i}=x1; 
end
e=t-x1;
F=F_cost(e);

%%敏感性反向传播
%最后一层的敏感性SM
s=cell(1,net_layer);
s{1,net_layer}=-sigmoid_d(x1)*e;
%倒退前面的敏感性Sk
for j=net_layer-1:-1:1;
    s{1,j}=sigmoid_d(x{1,j})*(2*index{u,j+1}*diag(x{1,j})+index{v,j+1})'*s{1,j+1};
%     s{1,j}=s{1,j+1}*(2*index{u,j+1}*diag(x{1,j})+index{v,j+1})*sigmoid_d(z{1,j});
end
%更新除第一层以外的权重u,v,b
for k=net_layer:-1:2;
    index{u,k}=index{u,k}-learningrate*s{1,k}*(x{1,k-1}.^2)';
    index{v,k}=index{v,k}-learningrate*s{1,k}*x{1,k-1}';
    index{b,k}=index{b,k}-learningrate*s{1,k};
end
%更新第一层的权重u,v,b
    index{u,1}=index{u,1}-learningrate*s{1,1}*(train_data.^2)';
    index{v,1}=index{v,1}-learningrate*s{1,1}*train_data';
    index{b,1}=index{b,1}-learningrate*s{1,1};

%%sigmod函数
function [x]= sigmoid(n)
	x =sigmf(n,[1,0]);

% function [a]= sigmoid(n)
% 	a = 1./(1+exp(-n));
    
%%代价函数F
function [a]= F_cost(n)
	a = 0.5.*(n*n');
%%sigmod函数求导
function [result]= sigmoid_d(a)%生成对角函数阵
	[r,c] = size(a);
	result = zeros(r,r);
	for i =1:r
		result(i,i) = (1-a(i))*a(i);%矩阵是按列取的
	end;
    
%%生成对角矩阵
function [result]= diag(a)%生成对角函数阵
	[r,c] = size(a);
	result = zeros(r,r);
	for i =1:r
		result(i,i) = a(i);%矩阵是按列取的
	end;






