mp=1.67e-27;
m=40*mp;
Q=1.6e-19;
epsilon=8.854e-12;
kq=Q^2/(4*pi*epsilon);
w1=2*pi*5e6;
w2=2*pi*0.2e6;
k1=m*w1^2;
k2=m*w2^2;
l2=(Q^2/(2*m*w2^2*pi*epsilon))^(1/3);

% p=100;
% q=100;
% E=zeros(p,q);
% 
% x1=0.5e-6;
% y1=5.5e-6;
% x2=([1:p]-5)*1e-7;
% y2=([1:q])*1e-7;
% 
% for i=1:p
%     for j=1:q
%         E(i,j)=kq/sqrt((x1-x2(i))^2+(y1-y2(j))^2)+k1/2*(x1^2+x2(i)^2)+k2/2*(y1^2+y2(j)^2);
%     end
% end
% surf(x2,y2,E);
Emin=kq/l2+2*k2/2*(l2/2)^2;
E1=[];
E=[];
for x1=-5e-6:0.05e-6:5e-6
    E1=[];
    for y1=-0.5e-4:0.1e-6:0.5e-4
        E0=kq/sqrt(x1^2+(y1+l2/2)^2)+k1/2*x1^2+k2/2*y1^2+k2/2*(l2/2)^2;
        if(E0>1e-21)
            E0=1e-21;
        end
           E1=[E1,E0];
                        
    end
    E=[E;E1];
    
end
x1=-5e-6:0.05e-6:5e-6;
y1=-0.5e-4:0.1e-6:0.5e-4;
ymax=[];

for i=1:length(x1)
    ymax1=-y1(1);
    for j=2:length(y1)-1
        if kq/sqrt(x1(i)^2+(y1(j)+l2/2)^2)+k1/2*x1(i)^2+k2/2*y1(j)^2>kq/sqrt(x1(i)^2+(y1(j-1)+l2/2)^2)+k1/2*x1(i)^2+k2/2*y1(j-1)^2 && kq/sqrt(x1(i)^2+(y1(j)+l2/2)^2)+k1/2*x1(i)^2+k2/2*y1(j)^2>kq/sqrt(x1(i)^2+(y1(j+1)+l2/2)^2)+k1/2*x1(i)^2+k2/2*y1(j+1)^2
            ymax1=y1(j);
        end
    end
    ymax=[ymax,ymax1];
end
surf(y1,x1,E);
imagesc(y1,x1,E);
plot(x1,ymax);
plot(x1,kq./sqrt(x1.^2+(ymax+l2/2).^2)+k1/2*x1.^2+k2/2*ymax.^2+k2/2*(l2/2)^2);