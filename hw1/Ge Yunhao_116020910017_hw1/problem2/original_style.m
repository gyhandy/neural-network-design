function [ w,b ] = original_style( training_set,study_rate )  
%training_set��һ��m*nά�������е�һ����y_i,ʣ�µ��е�x_i  
%ѡȡ��ʼֵw_0��b_0  
w=[1;1];  
b=-1;  
e=0;
a=0;
p=training_set(:,1:(size(training_set,2)-1));
t=training_set(:,size(training_set,2));
count=0;            %ÿһ����ȷ��������  
hold on;         
axis([-10 10 -10 10]); % ������������ָ��������
for j=1:1:size(p,1);% ����
plot(p(j,1),p(j,2),'r.','MarkerSize',10);
end
        
while count ~= size(training_set,1)  
    count=0;  
    %��ѵ������ѡȡ���ݣ�x_i,y_i��  
    for i=1:size(p,1) %���б�������ÿ��������ѵ�� 
        count = count+1;
        a=hardlim(w'*p(i,:)'+b);
        e=t(i,:)-a;
         if e~=0
           count=count-1;
         end  
        w = w + study_rate*e*p(i,:)';%����w
        b = b + study_rate*e;%����b
       hold on;
         if w(2,:)~=0;
            x=[-5:0.001:5];
            y=((-w(1,:))*x-b)/w(2,:);
%        x=[(-b)/w(1,:),0];
%        y=[0,(-b)/w(2,:)];
            plot(x,y);
            axis([-10 10 -10 10]); % ������������ָ��������
         end
    end
end 
    fprintf('w=''%3.2f\n',w);%���w
    fprintf('b=''%3.2f\n',b);%���b
    fprintf('count=''%3.2f\n',count);%�����ȷ�����            

end  