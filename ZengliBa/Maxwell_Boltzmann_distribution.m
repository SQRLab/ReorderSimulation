% define constant
mp=1.67e-27;
m1=40*mp;
m2=2*mp;
k=1.38e-23;
epsilon=8.854e-12;
miu=1.257e-6;
T=300;

% % define maxwell_boltzmann destribution
% fun=@(v) 4*pi*v.^2*(m2/(2*pi*k*T)).^(3/2).*exp(-m2*v.*v/(2*k*T));
% 
% t=1-integral(fun,0,1000) %%for T=300K, probability of v>1e4 is about 1e-16


% distribution for speed
% n=1e5;
% nv=zeros(1,n);
% m=0;
% s=0;
% while(m<n)
%     a=rand;
%     b=rand;
%     a=1e4*a;
%     s=s+1;
%     if b<4*pi*a^2*(m2/(2*pi*k*T))^(3/2)*exp(-m2*a*a/(2*k*T))
%         m=m+1;
%         nv(m)=a;
%     end
% end
% nvm=zeros(1,100);
% for i=1:n
%     for j=1:100
%         if nv(i)<j*100
%             nvm(j)=nvm(j)+1;
%             break;
%         end
%     end
% end
% t=1:100:10000;
% plot(t,nvm);
    

% define maxwell_boltzmann destribution

% 3D
v=0:1e4;
y=4*pi*v.^2*(m2/(2*pi*k*T)).^(3/2).*exp(-m2*v.*v/(2*k*T));
plot(v,y);

% 2D
v=0:1e4;
y=2*pi*v.*(m2/(2*pi*k*T)).^(2/2).*exp(-m2*v.*v/(2*k*T));
plot(v,y);

% 1D
v=0:1e4;
y=2*(m2/(2*pi*k*T)).^(1/2).*exp(-m2*v.*v/(2*k*T));
plot(v,y);

% distribution for velocity
n=1e3;
nv=zeros(1,n);
vx=zeros(1,n);
v2=zeros(1,n);
for i=1:n
    vi=normrnd(0,sqrt(k*T/m2),1,3);
    nv(i)=sqrt(vi(1)*vi(1)+vi(2)*vi(2)+vi(3)*vi(3));
    vx(i)=vi(1);
    v2(i)=sqrt(vi(1)^2+vi(2)^2);
end

% 3D
nvm=zeros(1,100);
for i=1:n
    for j=1:100
        if nv(i)<j*100
            nvm(j)=nvm(j)+1;
            break;
        end
    end
end

v=0:1e4;
y=4*pi*v.^2*(m2/(2*pi*k*T)).^(3/2).*exp(-m2*v.*v/(2*k*T));
plot(v,y);
hold on
t=100:100:10000;
scatter(t-50,nvm/n/100);
hold off

% 2D
nv2m=zeros(1,100);
for i=1:n
    for j=1:100
        if v2(i)<j*100
            nv2m(j)=nv2m(j)+1;
            break;
        end
    end
end

v=0:1e4;
y=2*pi*v.*(m2/(2*pi*k*T)).^(2/2).*exp(-m2*v.*v/(2*k*T));
plot(v,y);
hold on
t=100:100:10000;
scatter(t-50,nv2m/n/100);
hold off

% 1D
nvxm=zeros(1,100);
for i=1:n
    for j=1:100
        if abs(vx(i))<j*100
            nvxm(j)=nvxm(j)+1;
            break;
        end
    end
end
v=0:1e4;
y=2*(m2/(2*pi*k*T)).^(1/2).*exp(-m2*v.*v/(2*k*T));
plot(v,y);
hold on
t=100:100:10000;
scatter(t-50,nvxm/n/100);
hold off