% define constant
mp=1.67e-27;
m1=40*mp;
m2=2*mp;
k=1.38e-23;
epsilon=8.854e-12;
miu=1.257e-6;
T=300;
Q=1.6e-19;
lengthz=10;
lengthr=10;
% define maxwell_boltzmann destribution
fun=@(v) 4*pi*v.^2*(m2/(2*pi*k*T)).^(3/2).*exp(-m2*v.*v/(2*k*T));

p=[];
omegaz=[];
omegar=[];
vm=[];
Utot2=[];
d=[];

for i=1:lengthz
    wz=1e6*i/lengthz;
    for j=1:lengthr
        wr=10e6*j/lengthr;
        omegaz=[omegaz,wz];
        omegar=[omegar,wr];
        
% calculate U1        
syms a b
expr1=m1*wz^2*(a^2+b^2)+Q^2/(4*pi*epsilon)*(2/(b-a)+2/(b+a)+1/(2*a)+1/(2*b));
eqns1=[2*m1*wz^2*a+Q^2/(4*pi*epsilon)*(2/(b-a)^2-2/(b+a)^2-1/(2*a^2))==0, 2*m1*wz^2*b+Q^2/(4*pi*epsilon)*(-2/(b-a)^2-2/(b+a)^2-1/(2*b^2))==0];
vars=[a,b];
assume(vars,'positive');
[sola1, solb1]=vpasolve(eqns1,vars);
U1=subs(expr1,[a,b],[sola1,solb1]);

% calculate U2
expr2=m1*(wr^2*a^2+wz^2*b^2)+Q^2/(4*pi*epsilon)*(1/(2*a)+1/(2*b)+4/sqrt(a^2+b^2));
eqns2=[2*m1*wr^2*a+Q^2/(4*pi*epsilon)*(-1/(2*a^2)-4*a/(a^2+b^2)^(3/2))==0, 2*m1*wz^2*b+Q^2/(4*pi*epsilon)*(-1/(2*b^2)-4*b/(a^2+b^2)^(3/2))==0];
vars=[a,b];
assume(vars,'positive');
[sola, solb]=vpasolve(eqns2,vars);
U2=subs(expr2,[a,b],[sola,solb]);
U=U2-U1;
Utot2=[Utot2;U];
% d=[d;[sola1,solb1,sola,solb]];
%         if(U<0)
%          p=[p,1];
%         else 
%             vt=(m1+m2)/m2*sqrt(U/m1);
%             % debug    vt=1000;
%             vm=[vm,vt];
%             t=integral(fun,0,vt);
%             p=[p,1-t];
%         end
    end
end

% scatter3(omegaz,omegar,p);

% wz=1e6;
% wr=5e6;

% x=sola;
% y=solb;
% expr1=a+b+1;
% e1=subs(expr1,[a,b],[sola,solb])