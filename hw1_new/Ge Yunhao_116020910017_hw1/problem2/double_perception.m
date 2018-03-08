%author Yunhao Ge£¨¸ğÔÆğ©£©
%student number 116020910017
clear all;  
training_set=[1,1;0,2;3,1;2,-1;2,0;1,-2;-1,2;-2,1;-1,1]; 
lable=[0,0;0,0;0,0;0,1;0,1;0,1;1,0;1,0;1,0];
study_rate=1;  
[u,b]=learning_rule( training_set,lable,study_rate ); 