%author Yunhao Ge（葛云皓）
%student number 116020910017
function [ w,b ] = learning_rule( training_set,lable,study_rate )  
%training_set是一个m*n维矩阵，其中第一行是y_i,剩下的行的x_i  
%选取初始值w_0，b_0  
p=training_set;
t=lable;
for k=1:size(t,2)  
w=[0.1;0.2];  
b=-1;  
e=0;
a=0;
count=0;            %每一次正确分类点个数  
hold on;         
axis([-10 10 -10 10]); % 设置坐标轴在指定的区间
for j=1:1:size(p,1);% 画点
plot(p(j,1),p(j,2),'r.','MarkerSize',10);
end
 
while count ~= size(training_set,1)  
    count=0;  
    %在训练集中选取数据（x_i,y_i）  
    for i=1:size(p,1) %按行遍历，对每个数进行训练 
        count = count+1;
        a=hardlim(w'*p(i,:)'+b);
        e=t(i,k)-a;
         if e~=0
           count=count-1;
         end  
        w = w + study_rate*e*p(i,:)';%修正w
        b = b + study_rate*e;%修正b
       
    end
end
     if w(2,:)~=0;
            x=[-5:0.001:5];
            y=((-w(1,:))*x-b)/w(2,:);
%        x=[(-b)/w(1,:),0];
%        y=[0,(-b)/w(2,:)];
            plot(x,y);
            axis([-10 10 -10 10]); % 设置坐标轴在指定的区间
    fprintf('w=''%3.2f\n',w);%输出w
    fprintf('b=''%3.2f\n',b);%输出b
    fprintf('count=''%3.2f\n',count);%输出正确分类点
    hold on;
end        

end 