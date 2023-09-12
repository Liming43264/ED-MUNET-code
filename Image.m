%% 导入数据

% load('Z_element.mat');
% load('Z_node.mat');
% e2=1120; 
 

% load('GY_sigma_1_1.mat');
% data=GY_sigma;


element(:,5)=data(:,1);

% %%%%%%%%%%%%%%画分布图%%%%%%%%%%%%

for i=1:e2
   pp=node(element(i,2:4),2:3);
   patch(pp(:,1),pp(:,2),element(i,5));         %element(i,5)为三角单元赋值
end


axis([-0.125 0.125,-0.125 0.125])
axis equal;
axis equal;
grid on;
colorbar  
figure;

%%%%%%%%%%%%%%%%%%%%%%%%%%%画像素图%%%%%%%%%%%%%%%%%%%%%%%%%%

zhongxin=ones(e2,1);
for i=1:e2
    zhongxin(i,1)=i;
    zhongxin(i,2)=(node(element(i,2),2)+node(element(i,3),2)+node(element(i,4),2))/3;
    zhongxin(i,3)=(node(element(i,2),3)+node(element(i,3),3)+node(element(i,4),3))/3;
end

X =zhongxin(:,2:3);
Y_mua=element(:,5);
M=300;
size_DOI = 0.24;
% M = 100; % the square containing the object has a dimension of MxM
d = size_DOI/M; %the nearest distance of two cell centers
tx = ((-(M-1)/2):1:((M-1)/2))*d; % 1 x M
ty = (((M-1)/2):(-1):(-(M-1)/2))*d; % 1 x M
[xx1,y1] = meshgrid(tx,ty);  % M x M
XI = [xx1(:) y1(:)];  %xx1(:)是把数组的每列以左右的次序，手尾相连成一个长列数组，再组成一个二维数组，水平串联数组
YI_mua=ones(M,M);
YI_mua=griddatan(X,Y_mua,XI);
YI_mua=reshape(YI_mua,size(xx1));%把一维数组YI_mua重排成size(xx1)数组



pcolor(xx1,y1,real(YI_mua));%像素图
% pcolor(xx1,y1,imag(YI_mua));
shading interp;  %在网格片内采用颜色插值处理，得出表面图显得最光滑
colormap parula;   %画出的图形的着色   cool是冷色

C=colorbar;
Ticks=[0,0.35,0.5,0.75,1];
TickLabels=[0,0.35,0.5,0.75,1];
colorbar  %显示颜色条

