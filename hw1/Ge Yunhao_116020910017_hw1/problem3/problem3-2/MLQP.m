%Yunhao Ge
%�����ز���4����Ԫ���������������Ԫ
function  MLQP()
 clear all;  
 training_set=load('training_set.txt'); 
 train_num=size(training_set,1);
 train_label=training_set(:,end-1:end);   %ѵ���������
 train_data=training_set(:,1:end-2); %ѵ���������ԣ�
 p =train_data;%ѵ����������
 t =train_label;%��ʦ�ź�
  [train_num , input_scale]= size(p) ;%��ģ
  alpha  = 0.1;%ѧϰЧ��
  threshold = 0.3;%  �������� ��e^2 < threshold
  ud1=0;  ud2=0;
  wd1=0;  wd2=0; 
  bd1=0;  bd2=0;  
  e=0;
  circle_time =0;
  hidden_unitnum = 4; ;%���ز�ĵ�Ԫ��
  output_unitnum = 2;
  u1 = rand(hidden_unitnum,input_scale);%4����Ԫ��ÿ����Ԫ����input_scale������
  u2 = rand(output_unitnum,hidden_unitnum);%2����Ԫ��ÿ����Ԫ����4����
  w1 = rand(hidden_unitnum,input_scale);%4����Ԫ��ÿ����Ԫ����input_scale������
  w2 = rand(output_unitnum,hidden_unitnum);%2����Ԫ��ÿ����Ԫ����4������
  b1 = rand(hidden_unitnum,1);
  b2 = rand(output_unitnum,1);

  while 1
    accumulate_error = 0.0;
    circle_time = circle_time +1;
	
    for i=1:train_num
        %ǰ�򴫲�
        x0 = double ( p(i,:)'  );%��i������
        z1 = u1*x0.^2+w1*x0+b1;
        x1 = logsig(z1);%����������
        z2 = u2*x1.^2+w2*x1+b2;
        x2 = logsig(z2);%���������
         %���򴫲�������
        e = (t(i,:))'- x2;
                               
        s2 = -2*F(z2)*e; 	      
        s1 =  F(z1)*(2*u1*(sigmoid(z1))+w2)'*s2;      

        %�޸�Ȩֵ
        ud1 = alpha .* s1*(x0').^2;
        ud2 = alpha .* s2*(x1').^2;
        u1 =  u1 -ud1;
        u2 =  u2 -ud2;
        wd1 = alpha .* s1*x0';
        wd2 = alpha .* s2*x1';
        w1 =  w1 -wd1;
        w2 =  w2 -wd2;
        bd1 = alpha .* s1;
        bd2 = alpha .* s2;
        b1 = b1-bd1;
        b2 = b2-bd2;        
    end;%end of for
    if accumulate_error <= threshold| circle_time>10000  %then	
        	break;
    end;%end of if
  end;%end of while

disp(['accumulate_error = ',num2str( accumulate_error)] )	;
disp('------------');disp(circle_time )	

%----------------------------------------------------------
function [a]= sigmoid(n)
	a = 1./(1+exp(-n));
%----------------------------------------------------------
function [result]= F(a)
	[r,c] = size(a);
	result = zeros(r,r);
	for i =1:r
		result(i,i) = (1-a(i))*a(i);
	end;

