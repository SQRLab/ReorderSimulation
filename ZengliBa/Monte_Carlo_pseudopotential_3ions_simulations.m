% 3_D 3 ions pseudopotential Monte Carlo simulations
mp=1.67e-27;
m1=40*mp;
m2=2*mp;
k=1.38e-23;
epsilon=8.854e-12;
miu=1.257e-6;
T=300;
% nb=6;
% reorder=zeros(1,nb); 
nw1=10;
nw2=5;
w1=zeros(1,nw1);
w2=zeros(1,nw2);
reorder=zeros(nw1,nw2);
 
% distribution for velocity
n=1e3; %%%%%%%%%% number of samples
nv=zeros(1,n);
vi=zeros(n,3);
theta=zeros(1,n);

for i=1:n
%     vi=normrnd(0,sqrt(k*T/m2),1,2);
%     nv(i)=sqrt(vi(1)*vi(1)+vi(2)*vi(2));
    vix=normrnd(0,sqrt(k*T/m2));
    viy=normrnd(0,sqrt(k*T/m2));
    viz=normrnd(0,sqrt(k*T/m2));
    vitot=sqrt(vix^2+viy^2+viz^2);
    xprime=[vix,viy,viz]/vitot;
    yprime=[rand-0.5,rand-0.5,rand-0.5];
    yprime=yprime-dot(xprime,yprime)*xprime;
    yprime=yprime/norm(yprime);
    beta=asin(2*rand-1);
    vi(i,:)=2*m2/(m1+m2)*cos(beta)*vitot*(cos(beta)*xprime+sin(beta)*yprime);
end

% plot speed
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


% plot velocity
% nvxm=zeros(1,1000);
% for i=1:n
%     for j=1:1000
%         if abs(vix(i))<j*10
%             nvxm(j)=nvxm(j)+1;
%             break;
%         end
%     end
% end
% t=1:10:10000;
% plot(t,nvxm);

for p=1:nw1
    for q=1:nw2
        
    
s=0;
for j=1:fix(2*n/3)
% 3-D 3 particles
% 


% define parameters
q0=0.3034;
w1(p)=p*2*pi*1e6;
w2(q)=q*2*pi*0.2e6;
Q=1.6e-19;
kq=Q^2/(4*pi*epsilon);
k1=m1*w1(p)^2-m1*w2(q)^2/2;
k2=m1*w2(q)^2;
omega=2*sqrt(2)*w1(p)/q0;

% time scale
tau=1e-9;
N=1e5; %%%%%%%% time scale
T=N*tau;

% initial parameters
l=(5*Q*Q/(16*m1*w2(q)^2*pi*epsilon))^(1/3);
r10=[0,0,l];
r20=-r10;
r30=[0,0,0];
v10=vi(j,:);
v20=[0,0,0];
v30=[0,0,0];

r=zeros(N,3,3);
v=zeros(N,3,3);
t=zeros(1,N);
d12=zeros(1,N);
d23=zeros(1,N);
d31=zeros(1,N);
E=zeros(1,N);
% r=[];
% v=[];
% t=[];
% d=[];
a=[];
t(1)=0;
% format x(time,order of the particle,spacial dimension)

% calculation
r(1,1,:)=r10;
r(1,2,:)=r20;
r(1,3,:)=r30;
v(1,1,:)=v10;
v(1,2,:)=v20;
v(1,3,:)=v30;
d12(1)=sqrt((r(1,1,1)-r(1,2,1))^2+(r(1,1,2)-r(1,2,2))^2+(r(1,1,3)-r(1,2,3))^2);
d23(1)=sqrt((r(1,2,1)-r(1,3,1))^2+(r(1,2,2)-r(1,3,2))^2+(r(1,2,3)-r(1,3,3))^2);
d31(1)=sqrt((r(1,1,1)-r(1,3,1))^2+(r(1,1,2)-r(1,3,2))^2+(r(1,1,3)-r(1,3,3))^2);


E(1)=m1/2*(v(1,1,1)^2+v(1,1,2)^2+v(1,1,3)^2+v(1,2,1)^2+v(1,2,2)^2+v(1,2,3)^2+v(1,3,1)^2+v(1,3,2)^2+v(1,3,3)^2)+...
    k1/2*(r(1,1,1)^2+r(1,2,1)^2+r(1,3,1)^2+r(1,1,2)^2+r(1,2,2)^2+r(1,3,2)^2)+k2/2*(r(1,1,3)^2+r(1,2,3)^2+r(1,3,3)^2)+...
    kq*(1/d12(1)+1/d23(1)+1/d31(1));

for i=2:N
    t(i)=t(i-1)+tau;
    d12(i)=sqrt((r(i-1,1,1)-r(i-1,2,1))^2+(r(i-1,1,2)-r(i-1,2,2))^2+(r(i-1,1,3)-r(i-1,2,3))^2);
    d23(i)=sqrt((r(i-1,2,1)-r(i-1,3,1))^2+(r(i-1,2,2)-r(i-1,3,2))^2+(r(i-1,2,3)-r(i-1,3,3))^2);
    d31(i)=sqrt((r(i-1,1,1)-r(i-1,3,1))^2+(r(i-1,1,2)-r(i-1,3,2))^2+(r(i-1,1,3)-r(i-1,3,3))^2);
    a(1,1,1)=(-(w1(p)^2+w2(q)^2/2)/(q0/4)*cos(omega*t(i-1))+w2(q)^2/2)*r(i-1,1,1)+kq/m1*((r(i-1,1,1)-r(i-1,2,1))/(d12(i))^3+(r(i-1,1,1)-r(i-1,3,1))/(d31(i))^3);
    a(1,1,2)=((w1(p)^2+w2(q)^2/2)/(q0/4)*cos(omega*t(i-1))+w2(q)^2/2)*r(i-1,1,2)+kq/m1*((r(i-1,1,2)-r(i-1,2,2))/(d12(i))^3+(r(i-1,1,2)-r(i-1,3,2))/(d31(i))^3);
    a(1,1,3)=-w2(q)^2*r(i-1,1,3)+kq/m1*((r(i-1,1,3)-r(i-1,2,3))/(d12(i))^3+(r(i-1,1,3)-r(i-1,3,3))/(d31(i))^3);
    a(1,2,1)=(-(w1(p)^2+w2(q)^2/2)/(q0/4)*cos(omega*t(i-1))+w2(q)^2/2)*r(i-1,2,1)+kq/m1*((r(i-1,2,1)-r(i-1,1,1))/(d12(i))^3+(r(i-1,2,1)-r(i-1,3,1))/(d23(i))^3);
    a(1,2,2)=((w1(p)^2+w2(q)^2/2)/(q0/4)*cos(omega*t(i-1))+w2(q)^2/2)*r(i-1,2,2)+kq/m1*((r(i-1,2,2)-r(i-1,1,2))/(d12(i))^3+(r(i-1,2,2)-r(i-1,3,2))/(d23(i))^3);
    a(1,2,3)=-w2(q)^2*r(i-1,2,3)+kq/m1*((r(i-1,2,3)-r(i-1,1,3))/(d12(i))^3+(r(i-1,2,3)-r(i-1,3,3))/(d23(i))^3);
    a(1,3,1)=(-(w1(p)^2+w2(q)^2/2)/(q0/4)*cos(omega*t(i-1))+w2(q)^2/2)*r(i-1,3,1)+kq/m1*((r(i-1,3,1)-r(i-1,2,1))/(d23(i))^3+(r(i-1,3,1)-r(i-1,1,1))/(d31(i))^3);
    a(1,3,2)=((w1(p)^2+w2(q)^2/2)/(q0/4)*cos(omega*t(i-1))+w2(q)^2/2)*r(i-1,3,2)+kq/m1*((r(i-1,3,2)-r(i-1,2,2))/(d23(i))^3+(r(i-1,3,2)-r(i-1,1,2))/(d31(i))^3);
    a(1,3,3)=-w2(q)^2*r(i-1,3,3)+kq/m1*((r(i-1,3,3)-r(i-1,2,3))/(d23(i))^3+(r(i-1,3,3)-r(i-1,1,3))/(d31(i))^3);
    v(i,:,:)=v(i-1,:,:)+tau*a;
    r(i,:,:)=r(i-1,:,:)+(v(i,:,:)+v(i,:,:))*tau/2;
    E(i)=m1/2*(v(i,1,1)^2+v(i,1,2)^2+v(i,1,3)^2+v(i,2,1)^2+v(i,2,2)^2+v(i,2,3)^2+v(i,3,1)^2+v(i,3,2)^2+v(i,3,3)^2)+...
    k1/2*(r(i,1,1)^2+r(i,2,1)^2+r(i,3,1)^2+r(i,1,2)^2+r(i,2,2)^2+r(i,3,2)^2)+k2/2*(r(i,1,3)^2+r(i,2,3)^2+r(i,3,3)^2)+...
    kq*(1/d12(i)+1/d23(i)+1/d31(i));
end

% plot(t,E);

% plot(t,r(:,1,1));
% plot(t,r(:,1,2));
% % hold on;
% plot(t,r(:,2,1));
% plot(t,r(:,2,2));
% 
% plot(t,r(:,1,3));
% hold on;
% plot(t,r(:,2,3));
% hold off;

% plot(t,r(:,2,1));
% hold on;
% plot(t,r(:,1,1));
% hold off;
for k=1:N
    if r(k,1,3)<r(k,2,3)
        reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,1,3)<r(k,3,3)
        reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,3,3)<r(k,2,3)
         reorder(p,q)=reorder(p,q)+1;
        break;
    end
end
s=s+1;
end



for j=(fix(2*n/3)+1):n
% 3-D 3 particles
% 


% define parameters
q0=0.3034;
w1(p)=p*2*pi*1e6;
w2(q)=q*2*pi*0.2e6;
Q=1.6e-19;
kq=Q^2/(4*pi*epsilon);
k1=m1*w1(p)^2-m1*w2(q)^2/2;
k2=m1*w2(q)^2;
omega=2*sqrt(2)*w1(p)/q0;

% time scale
tau=1e-9;
N=1e5; %%%%%%%% time scale
T=N*tau;

% initial parameters
l=(5*Q*Q/(16*m1*w2(q)^2*pi*epsilon))^(1/3);
r10=[0,0,l];
r20=-r10;
r30=[0,0,0];
v10=[0,0,0];
v20=[0,0,0];
v30=vi(j,:);

r=zeros(N,3,3);
v=zeros(N,3,3);
t=zeros(1,N);
d12=zeros(1,N);
d23=zeros(1,N);
d31=zeros(1,N);
E=zeros(1,N);
% r=[];
% v=[];
% t=[];
% d=[];
a=[];
t(1)=0;
% format x(time,order of the particle,spacial dimension)

% calculation
r(1,1,:)=r10;
r(1,2,:)=r20;
r(1,3,:)=r30;
v(1,1,:)=v10;
v(1,2,:)=v20;
v(1,3,:)=v30;
d12(1)=sqrt((r(1,1,1)-r(1,2,1))^2+(r(1,1,2)-r(1,2,2))^2+(r(1,1,3)-r(1,2,3))^2);
d23(1)=sqrt((r(1,2,1)-r(1,3,1))^2+(r(1,2,2)-r(1,3,2))^2+(r(1,2,3)-r(1,3,3))^2);
d31(1)=sqrt((r(1,1,1)-r(1,3,1))^2+(r(1,1,2)-r(1,3,2))^2+(r(1,1,3)-r(1,3,3))^2);


E(1)=m1/2*(v(1,1,1)^2+v(1,1,2)^2+v(1,1,3)^2+v(1,2,1)^2+v(1,2,2)^2+v(1,2,3)^2+v(1,3,1)^2+v(1,3,2)^2+v(1,3,3)^2)+...
    k1/2*(r(1,1,1)^2+r(1,2,1)^2+r(1,3,1)^2+r(1,1,2)^2+r(1,2,2)^2+r(1,3,2)^2)+k2/2*(r(1,1,3)^2+r(1,2,3)^2+r(1,3,3)^2)+...
    kq*(1/d12(1)+1/d23(1)+1/d31(1));

for i=2:N
    t(i)=t(i-1)+tau;
    d12(i)=sqrt((r(i-1,1,1)-r(i-1,2,1))^2+(r(i-1,1,2)-r(i-1,2,2))^2+(r(i-1,1,3)-r(i-1,2,3))^2);
    d23(i)=sqrt((r(i-1,2,1)-r(i-1,3,1))^2+(r(i-1,2,2)-r(i-1,3,2))^2+(r(i-1,2,3)-r(i-1,3,3))^2);
    d31(i)=sqrt((r(i-1,1,1)-r(i-1,3,1))^2+(r(i-1,1,2)-r(i-1,3,2))^2+(r(i-1,1,3)-r(i-1,3,3))^2);
    a(1,1,1)=(-(w1(p)^2+w2(q)^2/2)/(q0/4)*cos(omega*t(i-1))+w2(q)^2/2)*r(i-1,1,1)+kq/m1*((r(i-1,1,1)-r(i-1,2,1))/(d12(i))^3+(r(i-1,1,1)-r(i-1,3,1))/(d31(i))^3);
    a(1,1,2)=((w1(p)^2+w2(q)^2/2)/(q0/4)*cos(omega*t(i-1))+w2(q)^2/2)*r(i-1,1,2)+kq/m1*((r(i-1,1,2)-r(i-1,2,2))/(d12(i))^3+(r(i-1,1,2)-r(i-1,3,2))/(d31(i))^3);
    a(1,1,3)=-w2(q)^2*r(i-1,1,3)+kq/m1*((r(i-1,1,3)-r(i-1,2,3))/(d12(i))^3+(r(i-1,1,3)-r(i-1,3,3))/(d31(i))^3);
    a(1,2,1)=(-(w1(p)^2+w2(q)^2/2)/(q0/4)*cos(omega*t(i-1))+w2(q)^2/2)*r(i-1,2,1)+kq/m1*((r(i-1,2,1)-r(i-1,1,1))/(d12(i))^3+(r(i-1,2,1)-r(i-1,3,1))/(d23(i))^3);
    a(1,2,2)=((w1(p)^2+w2(q)^2/2)/(q0/4)*cos(omega*t(i-1))+w2(q)^2/2)*r(i-1,2,2)+kq/m1*((r(i-1,2,2)-r(i-1,1,2))/(d12(i))^3+(r(i-1,2,2)-r(i-1,3,2))/(d23(i))^3);
    a(1,2,3)=-w2(q)^2*r(i-1,2,3)+kq/m1*((r(i-1,2,3)-r(i-1,1,3))/(d12(i))^3+(r(i-1,2,3)-r(i-1,3,3))/(d23(i))^3);
    a(1,3,1)=(-(w1(p)^2+w2(q)^2/2)/(q0/4)*cos(omega*t(i-1))+w2(q)^2/2)*r(i-1,3,1)+kq/m1*((r(i-1,3,1)-r(i-1,2,1))/(d23(i))^3+(r(i-1,3,1)-r(i-1,1,1))/(d31(i))^3);
    a(1,3,2)=((w1(p)^2+w2(q)^2/2)/(q0/4)*cos(omega*t(i-1))+w2(q)^2/2)*r(i-1,3,2)+kq/m1*((r(i-1,3,2)-r(i-1,2,2))/(d23(i))^3+(r(i-1,3,2)-r(i-1,1,2))/(d31(i))^3);
    a(1,3,3)=-w2(q)^2*r(i-1,3,3)+kq/m1*((r(i-1,3,3)-r(i-1,2,3))/(d23(i))^3+(r(i-1,3,3)-r(i-1,1,3))/(d31(i))^3);
    v(i,:,:)=v(i-1,:,:)+tau*a;
    r(i,:,:)=r(i-1,:,:)+(v(i,:,:)+v(i,:,:))*tau/2;
    E(i)=m1/2*(v(i,1,1)^2+v(i,1,2)^2+v(i,1,3)^2+v(i,2,1)^2+v(i,2,2)^2+v(i,2,3)^2+v(i,3,1)^2+v(i,3,2)^2+v(i,3,3)^2)+...
    k1/2*(r(i,1,1)^2+r(i,2,1)^2+r(i,3,1)^2+r(i,1,2)^2+r(i,2,2)^2+r(i,3,2)^2)+k2/2*(r(i,1,3)^2+r(i,2,3)^2+r(i,3,3)^2)+...
    kq*(1/d12(i)+1/d23(i)+1/d31(i));
end

% plot(t,E);

% plot(t,r(:,1,1));
% plot(t,r(:,1,2));
% % hold on;
% plot(t,r(:,2,1));
% plot(t,r(:,2,2));
% 
% plot(t,r(:,1,3));
% hold on;
% plot(t,r(:,2,3));
% hold off;

% plot(t,r(:,2,1));
% hold on;
% plot(t,r(:,1,1));
% hold off;
for k=1:N
    if r(k,1,3)<r(k,2,3)
        reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,1,3)<r(k,3,3)
        reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,3,3)<r(k,2,3)
         reorder(p,q)=reorder(p,q)+1;
        break;
    end
end
s=s+1;
end


    end
    
end