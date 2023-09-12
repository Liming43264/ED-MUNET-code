%% ��������

% load('Z_element.mat');
% load('Z_node.mat');
% e2=1120; 
 

% load('GY_sigma_1_1.mat');
% data=GY_sigma;


element(:,5)=data(:,1);

% %%%%%%%%%%%%%%���ֲ�ͼ%%%%%%%%%%%%

for i=1:e2
   pp=node(element(i,2:4),2:3);
   patch(pp(:,1),pp(:,2),element(i,5));         %element(i,5)Ϊ���ǵ�Ԫ��ֵ
end


axis([-0.125 0.125,-0.125 0.125])
axis equal;
axis equal;
grid on;
colorbar  
figure;

%%%%%%%%%%%%%%%%%%%%%%%%%%%������ͼ%%%%%%%%%%%%%%%%%%%%%%%%%%

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
XI = [xx1(:) y1(:)];  %xx1(:)�ǰ������ÿ�������ҵĴ�����β������һ���������飬�����һ����ά���飬ˮƽ��������
YI_mua=ones(M,M);
YI_mua=griddatan(X,Y_mua,XI);
YI_mua=reshape(YI_mua,size(xx1));%��һά����YI_mua���ų�size(xx1)����



pcolor(xx1,y1,real(YI_mua));%����ͼ
% pcolor(xx1,y1,imag(YI_mua));
shading interp;  %������Ƭ�ڲ�����ɫ��ֵ�����ó�����ͼ�Ե���⻬
colormap parula;   %������ͼ�ε���ɫ   cool����ɫ

C=colorbar;
Ticks=[0,0.35,0.5,0.75,1];
TickLabels=[0,0.35,0.5,0.75,1];
colorbar  %��ʾ��ɫ��

