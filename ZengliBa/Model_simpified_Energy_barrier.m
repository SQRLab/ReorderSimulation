mp=1.67e-27;
m=40*mp;
Q=1.6e-19;
epsilon=8.854e-12;
kq=Q^2/(4*pi*epsilon);


lengthz=5;
lengthr=10;
Emin=zeros(lengthz,lengthr);
Eminprime=zeros(lengthz,lengthr);
Ebarriermin=zeros(lengthz,lengthr);
for p=1:lengthz
    for q=1:lengthr

w1=2*pi*10e6*q/lengthr;
w2=2*pi*1e6*p/lengthz;
k1=m*w1^2;
k2=m*w2^2;
l2=(Q^2/(2*m*w2^2*pi*epsilon))^(1/3);
        
x1=-0.1e-4:0.5e-7:0.1e-4;
y1=-0.5e-4:0.1e-7:0.5e-4;
E=zeros(length(x1),length(y1));
Emin(p,q)=kq/l2+2*k2/2*(l2/2)^2;
for i=1:length(x1)
    for j=1:length(y1)
        E(i,j)=kq/sqrt(x1(i)^2+(y1(j)+l2/2)^2)+k1/2*x1(i)^2+k2/2*y1(j)^2+k2/2*(l2/2)^2;                                
    end       
end


ymax=zeros(1,length(x1));

for i=1:length(x1)
    ymax(i)=y1(1);
    for j=2:length(y1)-1
        if kq/sqrt(x1(i)^2+(y1(j)+l2/2)^2)+k1/2*x1(i)^2+k2/2*y1(j)^2>kq/sqrt(x1(i)^2+(y1(j-1)+l2/2)^2)+k1/2*x1(i)^2+k2/2*y1(j-1)^2 && kq/sqrt(x1(i)^2+(y1(j)+l2/2)^2)+k1/2*x1(i)^2+k2/2*y1(j)^2>kq/sqrt(x1(i)^2+(y1(j+1)+l2/2)^2)+k1/2*x1(i)^2+k2/2*y1(j+1)^2
            ymax(i)=y1(j);
        end
    end
end
Eprime=kq./sqrt(x1.^2+(ymax+l2/2).^2)+k1/2*x1.^2+k2/2*ymax.^2+k2/2*(l2/2)^2;
% surf(y1,x1,E);
% imagesc(y1,x1,E);
% plot(x1,ymax);
% hold on
% plot(x1,Eprime);
% hold off

for i=2:length(x1)-1
    if Eprime(i)<Eprime(i-1)&&Eprime(i)<Eprime(i+1)
        Eminprime(p,q)=Eprime(i);
    end
end
Ebarriermin(p,q)=Eminprime(p,q)-Emin(p,q);

        
    end
end

w1=(2*pi*10e6/lengthr)*(1:lengthr);
w2=(2*pi*1e6/lengthz)*(1:lengthz);
surf(w1/(2*pi),w2/(2*pi),Ebarriermin);
imagesc(w1/(2*pi),w2/(2*pi),Ebarriermin);