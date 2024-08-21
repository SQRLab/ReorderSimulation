% 2_D 4 ions monte Carlo simulations
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
vi=zeros(1,n);
theta=zeros(1,n);
beta=zeros(1,n);
for i=1:n
%     vi=normrnd(0,sqrt(k*T/m2),1,2);
%     nv(i)=sqrt(vi(1)*vi(1)+vi(2)*vi(2));
    vix=normrnd(0,sqrt(k*T/m2));
    viy=normrnd(0,sqrt(k*T/m2));
    vi(i)=sqrt(vix^2+viy^2);
    theta(i)=atan(viy/vix);
    beta(i)=asin(2*rand-1);
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
for j=1:500
% 2-D 3 particles
% 


% define parameters
w1(p)=p*2*pi*1e6;
w2(q)=q*2*pi*0.2e6;
Q=1.6e-19;
kq=Q^2/(4*pi*epsilon);
k1=m1*w1(p)^2;
k2=m1*w2(q)^2;

% time scale
tau=5e-11;
N=1e5; %%%%%%%% time scale
T=N*tau;

% initial parameters

l=(Q*Q/(4*m1*(w2(q))^2*pi*epsilon))^(1/3);
r10=[0,-1.4368*l];
r20=[0,-0.45438*l];
r30=[0,0.45438*l];
r40=[0,1.4368*l];

v10=2*m2/(m1+m2)*cos(beta(j))*vi(j)*[cos(theta(j)+beta(j)),sin(theta(j)+beta(j))];
v20=[0,0];
v30=[0,0];
v40=[0,0];



r=zeros(N,4,2);
v=zeros(N,4,2);
t=zeros(1,N);
d12=zeros(1,N);
d13=zeros(1,N);
d14=zeros(1,N);
d23=zeros(1,N);
d24=zeros(1,N);
d34=zeros(1,N);
E=zeros(1,N);
a=zeros(1,3,2);

r(1,1,:)=r10;
r(1,2,:)=r20;
r(1,3,:)=r30;
r(1,4,:)=r40;
v(1,1,:)=v10;
v(1,2,:)=v20;
v(1,3,:)=v30;
v(1,4,:)=v40;
d12(1)=sqrt((r(1,1,1)-r(1,2,1))^2+(r(1,1,2)-r(1,2,2))^2);
d13(1)=sqrt((r(1,1,1)-r(1,3,1))^2+(r(1,1,2)-r(1,3,2))^2);
d14(1)=sqrt((r(1,1,1)-r(1,4,1))^2+(r(1,1,2)-r(1,4,2))^2);
d23(1)=sqrt((r(1,2,1)-r(1,3,1))^2+(r(1,2,2)-r(1,3,2))^2);
d24(1)=sqrt((r(1,2,1)-r(1,4,1))^2+(r(1,2,2)-r(1,4,2))^2);
d34(1)=sqrt((r(1,3,1)-r(1,4,1))^2+(r(1,3,2)-r(1,4,2))^2);



E(1)=E(1)+m1/2*(v(1,1,1)^2+v(1,1,2)^2+v(1,2,1)^2+v(1,2,2)^2+v(1,3,1)^2+v(1,3,2)^2+v(1,4,1)^2+v(1,4,2)^2)+...
    k1/2*(r(1,1,1)^2+r(1,2,1)^2+r(1,3,1)^2+r(1,4,1)^2)+k2/2*(r(1,1,2)^2+r(1,2,2)^2+r(1,3,2)^2+r(1,4,2)^2)+...
    kq*(1/d12(1)+1/d13(1)+1/d14(1)+1/d23(1)+1/d24(1)+1/d34(1));


%%%% format r(time,order of the particle,spacial dimension)

for i=2:N
    t(i)=t(i-1)+tau;
    d12(i-1)=sqrt((r(i-1,1,1)-r(i-1,2,1))^2+(r(i-1,1,2)-r(i-1,2,2))^2);
    d13(i-1)=sqrt((r(i-1,1,1)-r(i-1,3,1))^2+(r(i-1,1,2)-r(i-1,3,2))^2);
    d14(i-1)=sqrt((r(i-1,1,1)-r(i-1,4,1))^2+(r(i-1,1,2)-r(i-1,4,2))^2);
    d23(i-1)=sqrt((r(i-1,2,1)-r(i-1,3,1))^2+(r(i-1,2,2)-r(i-1,3,2))^2);
    d24(i-1)=sqrt((r(i-1,2,1)-r(i-1,4,1))^2+(r(i-1,2,2)-r(i-1,4,2))^2);
    d34(i-1)=sqrt((r(i-1,3,1)-r(i-1,4,1))^2+(r(i-1,3,2)-r(i-1,4,2))^2);
    a(1,1,1)=-k1/m1*r(i-1,1,1)+kq/m1*((r(i-1,1,1)-r(i-1,2,1))/(d12(i-1))^3+(r(i-1,1,1)-r(i-1,3,1))/(d13(i-1))^3+(r(i-1,1,1)-r(i-1,4,1))/(d14(i-1))^3);
    a(1,1,2)=-k2/m1*r(i-1,1,2)+kq/m1*((r(i-1,1,2)-r(i-1,2,2))/(d12(i-1))^3+(r(i-1,1,2)-r(i-1,3,2))/(d13(i-1))^3+(r(i-1,1,2)-r(i-1,4,2))/(d14(i-1))^3);
    a(1,2,1)=-k1/m1*r(i-1,2,1)+kq/m1*((r(i-1,2,1)-r(i-1,1,1))/(d12(i-1))^3+(r(i-1,2,1)-r(i-1,3,1))/(d23(i-1))^3+(r(i-1,2,1)-r(i-1,4,1))/(d24(i-1))^3);
    a(1,2,2)=-k2/m1*r(i-1,2,2)+kq/m1*((r(i-1,2,2)-r(i-1,1,2))/(d12(i-1))^3+(r(i-1,2,2)-r(i-1,3,2))/(d23(i-1))^3+(r(i-1,2,2)-r(i-1,4,2))/(d24(i-1))^3);
    a(1,3,1)=-k1/m1*r(i-1,3,1)+kq/m1*((r(i-1,3,1)-r(i-1,1,1))/(d13(i-1))^3+(r(i-1,3,1)-r(i-1,2,1))/(d23(i-1))^3+(r(i-1,3,1)-r(i-1,4,1))/(d34(i-1))^3);
    a(1,3,2)=-k2/m1*r(i-1,3,2)+kq/m1*((r(i-1,3,2)-r(i-1,1,2))/(d13(i-1))^3+(r(i-1,3,2)-r(i-1,2,2))/(d23(i-1))^3+(r(i-1,3,2)-r(i-1,4,2))/(d34(i-1))^3);
    a(1,4,1)=-k1/m1*r(i-1,4,1)+kq/m1*((r(i-1,4,1)-r(i-1,1,1))/(d14(i-1))^3+(r(i-1,4,1)-r(i-1,2,1))/(d24(i-1))^3+(r(i-1,4,1)-r(i-1,3,1))/(d34(i-1))^3);
    a(1,4,2)=-k2/m1*r(i-1,4,2)+kq/m1*((r(i-1,4,2)-r(i-1,1,2))/(d14(i-1))^3+(r(i-1,4,2)-r(i-1,2,2))/(d24(i-1))^3+(r(i-1,4,2)-r(i-1,3,2))/(d34(i-1))^3);   
    v(i,:,:)=v(i-1,:,:)+tau*a;
    r(i,:,:)=r(i-1,:,:)+(v(i,:,:)+v(i-1,:,:))*tau/2;
    E(i)=m1/2*(v(i,1,1)^2+v(i,1,2)^2+v(i,2,1)^2+v(i,2,2)^2+v(i,3,1)^2+v(i,3,2)^2+v(i,4,1)^2+v(i,4,2)^2)+...
    k1/2*(r(i,1,1)^2+r(i,2,1)^2+r(i,3,1)^2+r(i,4,1)^2)+k2/2*(r(i,1,2)^2+r(i,2,2)^2+r(i,3,2)^2+r(i,4,2)^2)+...
    kq*(1/d12(i-1)+1/d13(i-1)+1/d14(i-1)+1/d23(i-1)+1/d24(i-1)+1/d34(i-1));

end

% plot(t,r(:,1,2));
% hold on
% plot(t,r(:,2,2));
% plot(t,r(:,3,2));
% hold off
% 
% plot(r(:,1,1),r(:,1,2));
plot(t,E);
% hold on


for k=1:N
    if r(k,1,2)>r(k,2,2)
        reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,1,2)>r(k,3,2)
        reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,1,2)>r(k,4,2)
         reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,2,2)>r(k,3,2)
         reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,2,2)>r(k,4,2)
         reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,3,2)>r(k,4,2)
         reorder(p,q)=reorder(p,q)+1;
        break;
    end
end
s=s+1;
end

for j=501:n
% 2-D 3 particles
% 


% define parameters
w1(p)=p*2*pi*1e6;
w2(q)=q*2*pi*0.2e6;
Q=1.6e-19;
kq=Q^2/(4*pi*epsilon);
k1=m1*w1(p)^2;
k2=m1*w2(q)^2;

% time scale
tau=5e-11;
N=1e5; %%%%%%%% time scale
T=N*tau;

% initial parameters

l=(Q*Q/(4*m1*(w2(q))^2*pi*epsilon))^(1/3);
r10=[0,-1.4368*l];
r20=[0,-0.45438*l];
r30=[0,0.45438*l];
r40=[0,1.4368*l];

v10=[0,0];
v20=2*m2/(m1+m2)*cos(beta(j))*vi(j)*[cos(theta(j)+beta(j)),sin(theta(j)+beta(j))];
v30=[0,0];
v40=[0,0];



r=zeros(N,4,2);
v=zeros(N,4,2);
t=zeros(1,N);
d12=zeros(1,N);
d13=zeros(1,N);
d14=zeros(1,N);
d23=zeros(1,N);
d24=zeros(1,N);
d34=zeros(1,N);
E=zeros(1,N);
a=zeros(1,3,2);

r(1,1,:)=r10;
r(1,2,:)=r20;
r(1,3,:)=r30;
r(1,4,:)=r40;
v(1,1,:)=v10;
v(1,2,:)=v20;
v(1,3,:)=v30;
v(1,4,:)=v40;
d12(1)=sqrt((r(1,1,1)-r(1,2,1))^2+(r(1,1,2)-r(1,2,2))^2);
d13(1)=sqrt((r(1,1,1)-r(1,3,1))^2+(r(1,1,2)-r(1,3,2))^2);
d12(1)=sqrt((r(1,1,1)-r(1,4,1))^2+(r(1,1,2)-r(1,4,2))^2);
d23(1)=sqrt((r(1,2,1)-r(1,3,1))^2+(r(1,2,2)-r(1,3,2))^2);
d24(1)=sqrt((r(1,2,1)-r(1,4,1))^2+(r(1,2,2)-r(1,4,2))^2);
d34(1)=sqrt((r(1,3,1)-r(1,4,1))^2+(r(1,3,2)-r(1,4,2))^2);



E(1)=E(1)+m1/2*(v(1,1,1)^2+v(1,1,2)^2+v(1,2,1)^2+v(1,2,2)^2+v(1,3,1)^2+v(1,3,2)^2+v(1,4,1)^2+v(1,4,2)^2)+...
    k1/2*(r(1,1,1)^2+r(1,2,1)^2+r(1,3,1)^2+r(1,4,1)^2)+k2/2*(r(1,1,2)^2+r(1,2,2)^2+r(1,3,2)^2+r(1,4,2)^2)+...
    kq*(1/d12(1)+1/d13(1)+1/d14(1)+1/d23(1)+1/d24(1)+1/d34(1));


%%%% format r(time,order of the particle,spacial dimension)

for i=2:N
    t(i)=t(i-1)+tau;
    d12(i-1)=sqrt((r(i-1,1,1)-r(i-1,2,1))^2+(r(i-1,1,2)-r(i-1,2,2))^2);
    d13(i-1)=sqrt((r(i-1,1,1)-r(i-1,3,1))^2+(r(i-1,1,2)-r(i-1,3,2))^2);
    d14(i-1)=sqrt((r(i-1,1,1)-r(i-1,4,1))^2+(r(i-1,1,2)-r(i-1,4,2))^2);
    d23(i-1)=sqrt((r(i-1,2,1)-r(i-1,3,1))^2+(r(i-1,2,2)-r(i-1,3,2))^2);
    d24(i-1)=sqrt((r(i-1,2,1)-r(i-1,4,1))^2+(r(i-1,2,2)-r(i-1,4,2))^2);
    d34(i-1)=sqrt((r(i-1,3,1)-r(i-1,4,1))^2+(r(i-1,3,2)-r(i-1,4,2))^2);
    a(1,1,1)=-k1/m1*r(i-1,1,1)+kq/m1*((r(i-1,1,1)-r(i-1,2,1))/(d12(i-1))^3+(r(i-1,1,1)-r(i-1,3,1))/(d13(i-1))^3+(r(i-1,1,1)-r(i-1,4,1))/(d14(i-1))^3);
    a(1,1,2)=-k2/m1*r(i-1,1,2)+kq/m1*((r(i-1,1,2)-r(i-1,2,2))/(d12(i-1))^3+(r(i-1,1,2)-r(i-1,3,2))/(d13(i-1))^3+(r(i-1,1,2)-r(i-1,4,2))/(d14(i-1))^3);
    a(1,2,1)=-k1/m1*r(i-1,2,1)+kq/m1*((r(i-1,2,1)-r(i-1,1,1))/(d12(i-1))^3+(r(i-1,2,1)-r(i-1,3,1))/(d23(i-1))^3+(r(i-1,2,1)-r(i-1,4,1))/(d24(i-1))^3);
    a(1,2,2)=-k2/m1*r(i-1,2,2)+kq/m1*((r(i-1,2,2)-r(i-1,1,2))/(d12(i-1))^3+(r(i-1,2,2)-r(i-1,3,2))/(d23(i-1))^3+(r(i-1,2,2)-r(i-1,4,2))/(d24(i-1))^3);
    a(1,3,1)=-k1/m1*r(i-1,3,1)+kq/m1*((r(i-1,3,1)-r(i-1,1,1))/(d13(i-1))^3+(r(i-1,3,1)-r(i-1,2,1))/(d23(i-1))^3+(r(i-1,3,1)-r(i-1,4,1))/(d34(i-1))^3);
    a(1,3,2)=-k2/m1*r(i-1,3,2)+kq/m1*((r(i-1,3,2)-r(i-1,1,2))/(d13(i-1))^3+(r(i-1,3,2)-r(i-1,2,2))/(d23(i-1))^3+(r(i-1,3,2)-r(i-1,4,2))/(d34(i-1))^3);
    a(1,4,1)=-k1/m1*r(i-1,4,1)+kq/m1*((r(i-1,4,1)-r(i-1,1,1))/(d14(i-1))^3+(r(i-1,4,1)-r(i-1,2,1))/(d24(i-1))^3+(r(i-1,4,1)-r(i-1,3,1))/(d34(i-1))^3);
    a(1,4,2)=-k2/m1*r(i-1,4,2)+kq/m1*((r(i-1,4,2)-r(i-1,1,2))/(d14(i-1))^3+(r(i-1,4,2)-r(i-1,2,2))/(d24(i-1))^3+(r(i-1,4,2)-r(i-1,3,2))/(d34(i-1))^3);   
    v(i,:,:)=v(i-1,:,:)+tau*a;
    r(i,:,:)=r(i-1,:,:)+(v(i,:,:)+v(i-1,:,:))*tau/2;
    E(i)=m1/2*(v(i,1,1)^2+v(i,1,2)^2+v(i,2,1)^2+v(i,2,2)^2+v(i,3,1)^2+v(i,3,2)^2+v(i,4,1)^2+v(i,4,2)^2)+...
    k1/2*(r(i,1,1)^2+r(i,2,1)^2+r(i,3,1)^2+r(i,4,1)^2)+k2/2*(r(i,1,2)^2+r(i,2,2)^2+r(i,3,2)^2+r(i,4,2)^2)+...
    kq*(1/d12(i-1)+1/d13(i-1)+1/d14(i-1)+1/d23(i-1)+1/d24(i-1)+1/d34(i-1));

end

% plot(t,r(:,1,2));
% hold on
% plot(t,r(:,2,2));
% plot(t,r(:,3,2));
% hold off
% 
% plot(r(:,1,1),r(:,1,2));
plot(t,E);
% hold on


for k=1:N
    if r(k,1,2)>r(k,2,2)
        reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,1,2)>r(k,3,2)
        reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,1,2)>r(k,4,2)
         reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,2,2)>r(k,3,2)
         reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,2,2)>r(k,4,2)
         reorder(p,q)=reorder(p,q)+1;
        break;
    elseif r(k,3,2)>r(k,4,2)
         reorder(p,q)=reorder(p,q)+1;
        break;
    end
end
s=s+1;
end


    end
    
end