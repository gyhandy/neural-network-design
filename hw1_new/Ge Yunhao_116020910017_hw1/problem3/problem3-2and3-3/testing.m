function predict =testing(index,test_data)
net_configure=[2,10,1];
net_layer=(size(net_configure,2)-1);
u=1;
v=2;
b=3;
x=test_data';
    for l=1:net_layer%����ÿһ����Ԫ
        z=(index{u,l}*x.^2)+(index{v,l}*x)+index{b,l};%zȫ�����棬������ʱ����
        x=sigmoid(z); %�����������
    end
    if x>0.5
        predict = 1;
    else
        predict = 0;
    end

%%sigmod����
function [x]= sigmoid(n)
	x =sigmf(n,[1,0]);
% function [a]= sigmoid(n)
% 	a = 1./(1+exp(-n));