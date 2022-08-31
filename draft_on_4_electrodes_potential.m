% Phi=V1*cos(omega*t)*(x^2-y^2)/r0^2+V2*(2*z^2-x^2-y^2)/(2*z0^2)
% U_{eff}=Q*(V2/z0^2)*z^2+(Q^2*V1^2/(4*m1*omega^2*r0^4)-Q*V1/(2*z0^2))*(x^2+y^2)

% % draft on 2D Paul trap
% % define constant
% mp=1.67e-27;
% m1=40*mp;
% m2=2*mp;
% k=1.38e-23;
% epsilon=8.854e-12;
% Q=1.6e-19;
% omega=2*pi*10e6;
% V1=500;
% 
% r0=2e-3;
% r1=[1e-5;1e-5];
% T=1e-4;
% N=1e5;
% tau=T/N;
% 
% t=zeros(1,N);
% r=zeros(2,N);
% v=zeros(2,N);
% r(:,1)=r1;
% v(:,1)=[0;0];
% 
% for i=1:N-1
%     t(i+1)=(i+1)*tau;
%     V=V1*cos(omega*t(i));
%     a=Q*V/(m1*r0^2)*[r(1,i);-r(2,i)];
%     a=Q*V/(m1*r0^2)*[r(1,i);-r(2,i)]+1e13*[r(1,i);-r(2,i)];
%     v(:,i+1)=v(:,i)+a*tau;
%     r(:,i+1)=r(:,i)+(v(:,i+1)+v(:,i))*tau/2;
% 
% end
% 
% plot(t,r(1,:));
% hold on
% plot(t,r(2,:));
% hold off
% 
% plot(r(1,:),r(2,:));




% draft on 3D Paul trap
% define constant
mp=1.67e-27;
m1=40*mp;
m2=2*mp;
k=1.38e-23;
epsilon=8.854e-12;
Q=1.6e-19;
omega=2*pi*10e6;
V1=500;
V2=10;
r0=2e-3;
z0=5e-2;
fr=sqrt(2*Q^2*V1^2/(m1^2*omega^2*r0^4)-Q*V2/(m1*z0^2))/(2*pi);
% fr=Q*V1/(sqrt(2)*omega*m1*r0^2)/(2*pi);
fz=sqrt(2*Q*V2/(m1*z0^2))/(2*pi);


T=1e-3;
N=1e5;
tau=T/N;

t=zeros(1,N);
r=zeros(3,N);
v=zeros(3,N);

r(:,1)=[1e-5;1e-5;1e-5];
v(:,1)=[0;0;0];

for i=1:N-1
    t(i+1)=(i+1)*tau;
    V=V1*cos(omega*t(i));
%     a=Q*V/(m1*r0^2)*[-r(1,i);r(2,i);0];
    a=2*Q*V/(m1*r0^2)*[-r(1,i);r(2,i);0]+Q*V2/(m1*z0^2)*[r(1,i);r(2,i);-2*r(3,i)];
    v(:,i+1)=v(:,i)+a*tau;
    r(:,i+1)=r(:,i)+(v(:,i+1)+v(:,i+1))*tau/2;

end

plot(t,r(1,:));
hold on
plot(t,r(2,:));
plot(t,r(3,:));
hold off

plot3(r(1,:),r(2,:),r(3,:));

q=4*Q*V1/(omega^2*m1*r0^2)

% y=fft(r(1,:));
% fs=1/tau;
% f=(0:length(y)-1)*fs/length(y);
% plot(f,abs(y))
% xlabel('Frequency (Hz)')
% ylabel('Magnitude')
% title('Magnitude')
% 
% n = length(r(1,:));                         
% fshift = (-n/2:n/2-1)*(fs/n);
% yshift = fftshift(y);
% plot(fshift,abs(yshift))
% xlabel('Frequency (Hz)')
% ylabel('Magnitude')







% % DRAFT FAILED
% % draft failed
% % define constant
% mp=1.67e-27;
% m1=40*mp;
% m2=2*mp;
% k=1.38e-23;
% epsilon=8.854e-12;
% Q=1.6e-19;
% omega=2*pi*1e5;
% 
% L=1e-2;
% r1=[1e-5,1e-5];
% 
% t=0:1e-7:1e-4;
% 
% Q1=Q*cos(omega*t);
% Q2=-Q*cos(omega*t);
% Q3=Q1;
% Q4=Q2;
% 
% d1=sqrt((r1(1)-L)^2+(r1(2)-L)^2);
% d2=sqrt((r1(1)+L)^2+(r1(2)-L)^2);
% d3=sqrt((r1(1)+L)^2+(r1(2)+L)^2);
% d4=sqrt((r1(1)-L)^2+(r1(2)+L)^2);
% 
% F1=[Q.*Q1/(4*pi*epsilon)*(r1(1)-L);Q.*Q1/(4*pi*epsilon)*(r1(2)-L)]/d1^3;
% plot(t,F1(1,:));
% hold on
% plot(t,F1(2,:));
% hold off
% 
% F2=[Q.*Q2/(4*pi*epsilon)*(r1(1)+L);Q.*Q2/(4*pi*epsilon)*(r1(2)-L)]/d2^3;
% plot(t,F2(1,:));
% hold on
% plot(t,F2(2,:));
% hold off
% 
% F3=[Q.*Q3/(4*pi*epsilon)*(r1(1)+L);Q.*Q3/(4*pi*epsilon)*(r1(2)+L)]/d3^3;
% plot(t,F3(1,:));
% hold on
% plot(t,F3(2,:));
% hold off
% 
% F4=[Q.*Q4/(4*pi*epsilon)*(r1(1)-L);Q.*Q4/(4*pi*epsilon)*(r1(2)+L)]/d4^3;
% plot(t,F4(1,:));
% hold on
% plot(t,F4(2,:));
% hold off
% 
% F=F1+F2+F3+F4;
% plot(t,F(1,:));
% hold on
% plot(t,F(2,:));
% hold off


% % draft failed
% % define constant
% mp=1.67e-27;
% m1=40*mp;
% m2=2*mp;
% k=1.38e-23;
% epsilon=8.854e-12;
% Q=1.6e-19;
% omega=2*pi*1e5;
% 
% L=1e-2;
% r1=[1e-5;1e-5];
% T=1e-4;
% N=1e5;
% tau=T/N;
% 
% t=zeros(1,N);
% r=zeros(2,N);
% v=zeros(2,N);
% r(:,1)=r1;
% 
% 
% % for i=1:N-1
% %     t(i+1)=(i+1)*tau;
% %     Q1=Q*cos(omega*t(i));
% %     Q2=-Q*cos(omega*t(i));
% %     Q3=Q1;
% %     Q4=Q2;
% %     d1=sqrt((r(i)-L)^2+(r(i)-L)^2);
% %     d2=sqrt((r(i)+L)^2+(r(i)-L)^2);
% %     d3=sqrt((r(i)+L)^2+(r(i)+L)^2);
% %     d4=sqrt((r(i)-L)^2+(r(i)+L)^2);
% %     F1=[Q*Q1/(4*pi*epsilon)*(r(i)-L);Q*Q1/(4*pi*epsilon)*(r(i)-L)]/d1^3;
% %     F2=[Q*Q2/(4*pi*epsilon)*(r(i)+L);Q*Q2/(4*pi*epsilon)*(r(i)-L)]/d2^3;
% %     F3=[Q*Q3/(4*pi*epsilon)*(r(i)+L);Q*Q3/(4*pi*epsilon)*(r(i)+L)]/d3^3;
% %     F4=[Q*Q4/(4*pi*epsilon)*(r(i)-L);Q*Q4/(4*pi*epsilon)*(r(i)+L)]/d4^3;
% %     F=F1+F2+F3+F4;
% %     a=F/m1;
% %     v(:,i+1)=v(:,i)+a*tau;
% %     r(:,i+1)=r(:,i)+(v(:,i+1)+v(:,i+1))*tau/2;
% % 
% % end
% 
% v(:,1)=[10,10];
% 
% for i=1:N-1
%     t(i+1)=(i+1)*tau;
%     Q1=Q*cos(omega*t(i));
%     Q2=-Q*cos(omega*t(i));
%     Q3=Q1;
%     Q4=Q2;
%     F=6e8*Q*Q1/(4*pi*epsilon*L^2)*[2*r(1,i);-2*r(2,i)];
%     a=F/m1;
%     v(:,i+1)=v(:,i)+a*tau;
%     r(:,i+1)=r(:,i)+(v(:,i+1)+v(:,i+1))*tau/2;
% 
% end
% 
% plot(t,r(1,:));
% hold on
% plot(t,r(2,:));
% hold off