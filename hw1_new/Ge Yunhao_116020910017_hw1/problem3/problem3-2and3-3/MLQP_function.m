function index=MLQP_function(train_data1,train_label1)
train_circle_number=1000;  
%ѵ�����޸�Ȩ��
net_configure=[2,10,1];%��������ṹ
learningrate=0.1;
%learningrate=0.2;
%learningrate=0.01;
train_number=size(train_data1,2);
index=net_init(net_configure);%����
for i=1:train_circle_number%ѭ������
    for j=1:train_number %������������
        train_data=train_data1(:,j);%��ȡһ��ѵ������
        train_lable=train_label1(:,j);%��ȡһ��ѵ����ǩ
        %��ÿ���������ѭ��ѵ��
        index=training(index,net_configure,train_data,train_lable,learningrate);%��ȡ������ݽ���ѧϰ���޸�Ȩ��
    end
end



