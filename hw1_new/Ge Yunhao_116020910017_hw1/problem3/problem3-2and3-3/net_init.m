%%初始化网络结构函数
function index=net_init(net_configure)
net_layer=(size(net_configure,2)-1);
index=cell(3,net_layer);
u=1;
v=2;
b=3;
for i=1:net_layer
    index{u,i}=rand(net_configure(1,i+1),net_configure(1,i));
    index{v,i}=rand(net_configure(1,i+1),net_configure(1,i));
    index{b,i}=rand(net_configure(1,i+1),1);
end





