clear all;  
training_set=[0,1,1;2,1,0;2,2,0;1,2,0];  
p=training_set(:,1:(size(training_set,2)-1));
hold on;
        axis([-5 5 -5 5]); % 设置坐标轴在指定的区间
         for j=1:1:size(p,1);% 画点
         plot(p(j,1),p(j,2),'r.','MarkerSize',10);
         end
        