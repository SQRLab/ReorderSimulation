mp=1.67e-27;
m=40*mp;
Q=1.6e-19;
epsilon=8.854e-12;
kq=Q^2/(4*pi*epsilon);
w1=2*pi*5e6;
w2=2*pi*0.2e6;
k1=m*w1^2/(2*pi);
k2=m*w2^2/(2*pi);


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
E1=[];
E=[];
for x1=-10e-6:0.1e-6:10e-6
    for x2=-10e-6:0.1e-6:10e-6
        E1=[E1,kq/sqrt((x1-x2)^2)+k1/2*(x1^2+x2^2)];
    end
    E=[E;E1];
    E1=[];
end
x1=-10e-6:0.1e-6:10e-6;
x2=-10e-6:0.1e-6:10e-6;
surf(x1,x2,E);