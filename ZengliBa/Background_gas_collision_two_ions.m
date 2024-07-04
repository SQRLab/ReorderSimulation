% define constant
mp=1.67e-27;
m1=40*mp;
m2=2*mp;
k=1.38e-23;
epsilon=8.854e-12;
miu=1.257e-6;
T=300;
Q=1.6e-19;
lengthz=5;
lengthr=10;
% define maxwell_boltzmann destribution
fun=@(v) 4*pi*v.^2*(m2/(2*pi*k*T)).^(3/2).*exp(-m2*v.*v/(2*k*T));

p=[];
omegaz=[];
omegar=[];
vm=[];
Utot=[];
Ebarrier=zeros(lengthz,lengthr);

% % the former one using function scatter(line21-line47)
% for i=1:lengthz
%     wz=1e6*i/lengthz;
%     for j=1:lengthr
%         wr=10e6*j/lengthr;
%         omegaz=[omegaz,wz];
%         omegar=[omegar,wr];
%         a=(Q^2/(2*m1*wz^2*pi*epsilon))^(1/3);
%         U1=m1*wz^2*a^2/4+Q^2/(4*pi*epsilon*a);
%         b=(Q^2/(2*m1*wr^2*pi*epsilon))^(1/3);
%         U2=m1*wr^2*b^2/4+Q^2/(4*pi*epsilon*b);
%         % debug U=1;
%         U=U2-U1;
%         Utot=[Utot;[U,U1,U2]];
%         if(U<0)
%          p=[p,1];
%         else 
%             vt=(m1+m2)/m2*sqrt(U/m1);
%             % debug    vt=1000;
%             vm=[vm,vt];
%             t=integral(fun,0,vt);
%             p=[p,1-t];
%         end
%     end
% end
% 
% scatter3(omegaz,omegar,p);

% the latter one using function surf and imagesc
for i=1:lengthz
    wz=2*pi*1e6*i/lengthz;
    omegaz=[omegaz,wz];
    omegar=(1:lengthr)*2*pi*10e6/lengthr;
    pj=[];
    for j=1:lengthr
        wr=2*pi*10e6*j/lengthr;
%         omegar=[omegar,wr];
        a=(Q^2/(2*m1*wz^2*pi*epsilon))^(1/3);
        U1=m1*wz^2*a^2/4+Q^2/(4*pi*epsilon*a);
        b=(Q^2/(2*m1*wr^2*pi*epsilon))^(1/3);
        U2=m1*wr^2*b^2/4+Q^2/(4*pi*epsilon*b);
        % debug U=1;
        U=U2-U1;
        Utot=[Utot;[U,U1,U2]];
        Ebarrier(i,j)=U;
        if(U<0)
         pj=[pj;1];
        else 
            vt=(m1+m2)/m2*sqrt(U/m1);
            % debug    vt=1000;
            vm=[vm,vt];
            t=integral(fun,0,vt);
            pj=[pj;1-t];
        end
    end
    p=[p,pj];
end

surf(omegar/(2*pi),omegaz/(2*pi),Ebarrier);
imagesc(omegar/(2*pi),omegaz/(2*pi),Ebarrier);

surf(omegaz/(2*pi),omegar/(2*pi),p);
imagesc(omegaz/(2*pi),omegar/(2*pi),p);
