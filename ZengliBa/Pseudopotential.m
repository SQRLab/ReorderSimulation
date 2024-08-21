% draft on 3D Paul trap
% define constant
mp=1.67e-27;
m1=40*mp;
m2=2*mp;
k=1.38e-23;
epsilon=8.854e-12;
Q=1.6e-19;
q0=0.3034;

% omega=2*pi*10e6;
% V1=500;
% V2=10;
% r0=2e-3;
% z0=5e-2;
% fr=sqrt(2*Q^2*V1^2/(m1^2*omega^2*r0^4)-Q*V2/(m1*z0^2))/(2*pi);
% fz=sqrt(2*Q*V2/(m1*z0^2))/(2*pi);

w1=2*pi*fr;
w2=2*pi*fz;
omega=2*sqrt(2)*w1/q0;


tau=1e-8;
N=1e5;
T=N*tau;

t=zeros(1,N);
r=zeros(3,N);
v=zeros(3,N);

r(:,1)=[1e-5;1e-5;1e-5];
v(:,1)=[0;0;0];

for i=1:N-1
    t(i+1)=(i+1)*tau;
    V=V1*cos(omega*t(i));
%     a=sqrt(2)*w1*omega*cos(omega*t(i))*[-r(1,i);r(2,i);0]+w2^2/2*[r(1,i);r(2,i);-2*r(3,i)];
%     a=(w1^2+w2^2/2)*m1*omega^2*r0^2/(Q*V1)*cos(omega*t(i))*[-r(1,i);r(2,i);0]+w2^2/2*[r(1,i);r(2,i);-2*r(3,i)];
    a=(w1^2+w2^2/2)/(q0/4)*cos(omega*t(i))*[-r(1,i);r(2,i);0]+w2^2/2*[r(1,i);r(2,i);-2*r(3,i)];
    v(:,i+1)=v(:,i)+a*tau;
    r(:,i+1)=r(:,i)+(v(:,i+1)+v(:,i+1))*tau/2;

end

plot(t,r(1,:));
hold on
plot(t,r(2,:));
plot(t,r(3,:));
hold off

plot3(r(1,:),r(2,:),r(3,:));