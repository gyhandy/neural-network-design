function index=MLQP_function(train_data1,train_label1)
train_circle_number=1000;  
%训练并修改权重
net_configure=[2,10,1];%创建网络结构
learningrate=0.1;
%learningrate=0.2;
%learningrate=0.01;
train_number=size(train_data1,2);
index=net_init(net_configure);%参数
for i=1:train_circle_number%循环次数
    for j=1:train_number %遍历所有样本
        train_data=train_data1(:,j);%读取一个训练样本
        train_lable=train_label1(:,j);%读取一个训练标签
        %对每个输入进行循环训练
        index=training(index,net_configure,train_data,train_lable,learningrate);%对取入的数据进行学习并修改权重
    end
end



