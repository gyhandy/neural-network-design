clc;
clear all;
%��ȡ���ݲ�������
twospiraltrain=load('two_spiral_train.txt');
train_number=size(twospiraltrain,1);%ѵ�����ݸ���                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
train_label1=twospiraltrain(:,end)';   %ѵ���������
train_data1=twospiraltrain(:,1:end-1)'; %ѵ������

posiNum=find(train_label1(:)==1);
positrain_data = train_data1(:,posiNum);%�ҵ�������
nagNum=find(train_label1(:)==0);
nagtrain_data = train_data1(:,nagNum);%�ҵ�������

%%����a b ��Ϊ1��c d ��Ϊ0
% ������ab��1��
posiSum=size(positrain_data,2);
rand_posi=randperm(posiSum,posiSum);
a_num = rand_posi(:,1:size(rand_posi,2)/2);%��һ��a
a = positrain_data(:,a_num)';
a_lable = ones(size(a,1),1);
b_num = rand_posi(:,size(rand_posi,2)/2+1:size(rand_posi,2));%�ڶ���b
b = positrain_data(:,b_num)';
b_lable = ones(size(b,1),1);
% ������cd��2��
nagSum=size(nagtrain_data,2);
rand_nag=randperm(nagSum,nagSum);
c_num = rand_nag(:,1:size(rand_nag,2)/2);%��һ��a
c = nagtrain_data(:,c_num)';
c_lable = zeros(size(c,1),1);
d_num = rand_nag(:,size(rand_nag,2)/2+1:size(rand_nag,2));%�ڶ���b
d = nagtrain_data(:,d_num)';
d_lable = zeros(size(d,1),1);
%��ͼ
% figure;
% for i=1:1:size(a,1)
%     hold on;
%     plot(a(i,1),a(i,2),'ro');
%     axis([-4 4 -4 4]);
% end
% figure;
% for i=1:1:size(b,1)
%     hold on;
%     plot(b(i,1),b(i,2),'r+');
%     axis([-4 4 -4 4]);
% end
% figure;
% for i=1:1:size(c,1)
%      hold on;
%      plot(c(i,1),c(i,2),'go');
%      axis([-4 4 -4 4]);
% end
% figure;
% for i=1:1:size(a,1)
%      hold on;
%      plot(d(i,1),d(i,2),'g+');
%      axis([-4 4 -4 4]);
% end
%%
%�ֱ�ѧϰ[ac][ad][bc][bd]
%[ac]
ac=[a;c]';
ac_lable=[a_lable;c_lable]';
index_ac= MLQP_function(ac,ac_lable);
%[ad]
ad=[a;d]';
ad_lable=[a_lable;d_lable]';
index_ad= MLQP_function(ad,ad_lable);
%[bc]
bc=[b;c]';
bc_lable=[b_lable;c_lable]';
index_bc= MLQP_function(bc,bc_lable);
%[bd]
bd=[b;d]';
bd_lable=[b_lable;d_lable]';
index_bd= MLQP_function(bd,bd_lable);

%%
%��ȡ���Լ�
% twospiraltest=load('two_spiral_test.txt');
% test_number=size(twospiraltrain,1);%�������ݸ���
% test_label1=twospiraltest(:,end);   %�������ݱ�ǩ
% test_data1=twospiraltest(:,1:end-1); %��������
% predict=zeros(test_number,size(test_label1,2));


for x=-4:0.1:4;
    for y=-4:0.1:4;
        bundary_data=[x,y];
  
       
        %��ÿ��������в���
        %�ֱ����[ac][ad][bc][bd]
        pred_ac=testing(index_ac,bundary_data);%�Բ������ݽ��в��Բ����ؽ��
        pred_ad=testing(index_ad,bundary_data);
        pred_bc=testing(index_bc,bundary_data);
        pred_bd=testing(index_bd,bundary_data);
        
        %min(1)[ac,ad]
        if pred_ac==1 && pred_ad==1;
           min_acad=1;
        else
           min_acad=0;
        end
        %min(1)[bc,bd]
        if pred_bc==1 && pred_bd==1;
           min_bcbd=1;
        else
           min_bcbd=0;
        end
         %max(0) abcd
         if min_acad==0 && min_bcbd==0;
           max_abcd=0;
        else
           max_abcd=1;
        end
    if max_abcd==1;
        hold on;
        plot(bundary_data(1,1),bundary_data(1,2),'ro');
    else
        hold on;
        plot(bundary_data(1,1),bundary_data(1,2),'bo');
    end
end
        
        
end 

        
        
        
 
        
%         for j=1:test_number %������������
%         test_data=test_data1(j,:);%��ȡһ����������
%         test_lable=test_label1(j,:);%��ȡһ�����Ա�ǩ
%        
%         %��ÿ��������в���
%         %�ֱ����[ac][ad][bc][bd]
%         pred_ac=testing(index_ac,test_data,test_lable);%�Բ������ݽ��в��Բ����ؽ��
%         pred_ad=testing(index_ad,test_data,test_lable);
%         pred_bc=testing(index_bc,test_data,test_lable);
%         pred_bd=testing(index_bd,test_data,test_lable);
%         
%         %min(1)[ac,ad]
%         if pred_ac==1 && pred_ad==1;
%            min_acad=1;
%         else
%            min_acad=0;
%         end
%         %min(1)[bc,bd]
%         if pred_bc==1 && pred_bd==1;
%            min_bcbd=1;
%         else
%            min_bcbd=0;
%         end
%          %max(0) abcd
%          if min_acad==0 && pred_bd==0;
%            max_abcd=0;
%         else
%            max_abcd=1;
%         end
%          
%         predict(j,:)=max_abcd;
%  end
% 
%  
%  %����ѵ�����ͼ��
% %  figure;
% % for z=1:1:test_number
% %     if predict(z,:)==1;
% %         hold on;
% %         plot(test_data1(z,1),test_data1(z,2),'ro');
% %     else
% %         hold on;
% %         plot(test_data1(z,1),test_data1(z,2),'bo');
% %     end
% % end
%  
% 
% 
% 
% 
