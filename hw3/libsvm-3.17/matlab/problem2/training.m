%ǰ�򴫵����磬�õ�������e�������ۺ���F
%ȫ���þ�������
function [index] =training(index,net_configure,train_data,train_lable,learningrate)%����ȡ
net_layer=(size(net_configure,2)-1);
u=1;
v=2;
b=3;
x1=train_data;%ȡһ������
t=train_lable;%ȡһ����ǩ
z=cell(1,net_layer);%zÿһ����Ԫ���У�������Ŀ��һ��һ����������cell
x=cell(1,net_layer);
for i=1:net_layer%����ÿһ����Ԫ
    z1=(index{u,i}*x1.^2)+(index{v,i}*x1)+index{b,i};%zȫ�����棬������ʱ����
    x1=sigmoid(z1); %�����������
    z{1,i}=z1;
    x{1,i}=x1; 
end
e=t-x1;
F=F_cost(e);

%%�����Է��򴫲�
%���һ���������SM
s=cell(1,net_layer);
s{1,net_layer}=-sigmoid_d(x1)*e;
%����ǰ���������Sk
for j=net_layer-1:-1:1;
    s{1,j}=sigmoid_d(x{1,j})*(2*index{u,j+1}*diag(x{1,j})+index{v,j+1})'*s{1,j+1};
%     s{1,j}=s{1,j+1}*(2*index{u,j+1}*diag(x{1,j})+index{v,j+1})*sigmoid_d(z{1,j});
end
%���³���һ�������Ȩ��u,v,b
for k=net_layer:-1:2;
    index{u,k}=index{u,k}-learningrate*s{1,k}*(x{1,k-1}.^2)';
    index{v,k}=index{v,k}-learningrate*s{1,k}*x{1,k-1}';
    index{b,k}=index{b,k}-learningrate*s{1,k};
end
%���µ�һ���Ȩ��u,v,b
    index{u,1}=index{u,1}-learningrate*s{1,1}*(train_data.^2)';
    index{v,1}=index{v,1}-learningrate*s{1,1}*train_data';
    index{b,1}=index{b,1}-learningrate*s{1,1};

%%sigmod����
function [x]= sigmoid(n)
	x =sigmf(n,[1,0]);

% function [a]= sigmoid(n)
% 	a = 1./(1+exp(-n));
    
%%���ۺ���F
function [a]= F_cost(n)
	a = 0.5.*(n*n');
%%sigmod������
function [result]= sigmoid_d(a)%���ɶԽǺ�����
	[r,c] = size(a);
	result = zeros(r,r);
	for i =1:r
		result(i,i) = (1-a(i))*a(i);%�����ǰ���ȡ��
	end;
    
%%���ɶԽǾ���
function [result]= diag(a)%���ɶԽǺ�����
	[r,c] = size(a);
	result = zeros(r,r);
	for i =1:r
		result(i,i) = a(i);%�����ǰ���ȡ��
	end;






