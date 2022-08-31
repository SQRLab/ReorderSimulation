% need to load simulation result first

vp=zeros(10,5);
Ebarrier=zeros(10,5);
for i=1:10
    for j=1:5
        p0=reorder(i,j)/1000;
% define constant
mp=1.67e-27;
m1=40*mp;
m2=2*mp;
k=1.38e-23;
epsilon=8.854e-12;
miu=1.257e-6;
T=300;
Q=1.6e-19;
lengthz=100;
lengthr=100;
% define maxwell_boltzmann destribution
fun=@(v) 2*pi*v.*(m2/(2*pi*k*T)).*exp(-m2*v.*v/(2*k*T));

vp1=0;
vp2=1e4;
while vp2-vp1>1e-3
    vp0=(vp1+vp2)/2;
    if integral(fun,0,vp0)==1-p0
        break;
    elseif integral(fun,0,vp0)>1-p0
        vp2=vp0;
    else
        vp1=vp0;
    end
end

vp(i,j)=vp0;
Ebarrier(i,j)=m1/2*vp(i,j)^2*(2*m2)^2/(m1+m2)^2;
    end
end
surf(w2/(2*pi),w1/(2*pi),Ebarrier);
imagesc(w2/(2*pi),w1/(2*pi),Ebarrier);