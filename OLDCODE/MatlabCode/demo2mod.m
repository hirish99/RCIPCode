  function demo2mod
% ********************************************************
% *** Solves (4) and computes q of (6) with basic RCIP   *
% *** Last changed 2018-01-28 by Johan Helsing           *
% ********************************************************
  close all
  format long
  format compact
% *** User specified quantities ***************

  x = [];
  array = [];
  for k=10:1:40
     array = [array, (log10((ErrorCompDL(10, k))))];
     x = [x, log10(k)];
  end
  #ErrorCompDL(10, 4)
  
  p= polyfit(x,array,1);
  figure
  plot(x, polyval(p,x));
  hold on
  scatter(x,array);
  title(num2str(p));
  hold off
  
  function error=ErrorCompOld(npan,nsub)
    lambda=0.999;  % parameter lambda
    %lambda=1;
    theta=pi/2;    % corner opening angle
    evec=1;        % a unit vector 
    qref=1.1300163213105365; % reference solution
  % *********************************************
    T=Tinit16;
    W=Winit16;
  %
  % *** Pbc and PWbc prolongation matrices ***
    [IP,IPW]=IPinit(T,W);
    Pbc =blkdiag(eye(16),IP ,IP ,eye(16));
    PWbc=blkdiag(eye(16),IPW,IPW,eye(16));
    %disp(['check_PWTP = ',num2str(norm(PWbc'*Pbc-eye(64)))])
  %
  % *** Panels, discretization points, and weights ***
    sinter=linspace(0,1,npan+1)';
    sinterdiff=ones(npan,1)/npan;
    [z,zp,zpp,nz,w,wzp,np]=zinit(theta,sinter,sinterdiff,T,W,npan);
    %disp(['arclength = ',num2str(sum(abs(wzp)),16)])
    %disp(['area      = ',num2str(0.5*imag(conj(z).'*wzp),15)])
  %
  % *** The K_coa^\circ matrix is set up ***
    Kcirc=MAinit(z,zp,zpp,nz,w,wzp,np);
    %KcircD = MAinitDL(z,zp,zpp,nz,w,wzp,np);
    %Kcirc=KcircD;
    
    starind=[np-31:np 1:32];
    Kcirc(starind,starind)=zeros(64);
  %
  % *** Recursion for the R matrix ***
    R=speye(np);
    R(starind,starind)=Rcomp(theta,lambda,T,W,Pbc,PWbc,nsub,npan);
  %
    conj(evec)
  % *** Solving main linear system ***
    rhs=2*lambda*real(conj(evec)*nz);
    
    
    %test_charge = complex(-0.25,0.4);
    %rhs=2*GetBC(test_charge, z);
    
    [rhotilde,it]=myGMRESR(lambda*Kcirc,R,rhs,np,100,eps);
    disp(['GMRES iter RCIP = ',num2str(it)])
  %
  % *** Post processing ***
    rhohat=R*rhotilde;
    zeta=real(conj(evec)*z).*abs(wzp);
    
    %target_complex = complex(0.01, 0)
    %zeta = MAOffBoundary(z,zp,zpp,nz,w,wzp,np,target_complex);
    
    q=rhohat'*zeta;
    %qref=G(target_complex, test_charge);
    error = (abs(qref-q)/abs(qref));
    %error = abs(q)


    
  function error=ErrorCompDL(npan,nsub)
    lambda=1.0000;  % parameter lambda
    %lambda=1;
    theta=pi/2;    % corner opening angle
    evec=1;        % a unit vector 
    qref=1.1300163213105365; % reference solution
  % *********************************************
    T=Tinit16;
    W=Winit16;
  %
  % *** Pbc and PWbc prolongation matrices ***
    [IP,IPW]=IPinit(T,W);
    Pbc =blkdiag(eye(16),IP ,IP ,eye(16));
    PWbc=blkdiag(eye(16),IPW,IPW,eye(16));
    %disp(['check_PWTP = ',num2str(norm(PWbc'*Pbc-eye(64)))])
  %
  % *** Panels, discretization points, and weights ***
    sinter=linspace(0,1,npan+1)';
    sinterdiff=ones(npan,1)/npan;
    [z,zp,zpp,nz,w,wzp,np]=zinit(theta,sinter,sinterdiff,T,W,npan);
    %disp(['arclength = ',num2str(sum(abs(wzp)),16)])
    %disp(['area      = ',num2str(0.5*imag(conj(z).'*wzp),15)])
  %
  % *** The K_coa^\circ matrix is set up ***
    %Kcirc=MAinit(z,zp,zpp,nz,w,wzp,np);
    KcircD = MAinitDL(z,zp,zpp,nz,w,wzp,np);
    Kcirc=KcircD;
    
    starind=[np-31:np 1:32];
    Kcirc(starind,starind)=zeros(64);
  %
  % *** Recursion for the R matrix ***
    R=speye(np);
    R(starind,starind)=RcompNew(theta,T,W,Pbc,PWbc,nsub,npan);
  %
    %conj(evec)
  % *** Solving main linear system ***
    %rhs=2*lambda*real(conj(evec)*nz);
    
    test_charge = complex(-0.25,0.4);
    rhs=GetBC(test_charge, z);
    
    [rhotilde,it]=myGMRESR(lambda*Kcirc,R,rhs,np,100,eps);
    disp(['GMRES iter RCIP = ',num2str(it)])
  %
  % *** Post processing ***
    rhohat=R*rhotilde;
    %zeta=real(conj(evec)*z).*abs(wzp);
    
    target_complex = complex(0.01, 0);
    zeta = MAOffBoundary(z,zp,zpp,nz,w,wzp,np,target_complex);
    
    %q=rhohat'*zeta
    q=zeta*rhohat;
    qref=G(target_complex, test_charge);
    error = (abs(qref-q)/abs(qref));
    #error = abs(q)
  
 
  function O=G(x, y)
    O = (-1/(2*pi)) * log(abs(x-y));
 
 
  function P=GetBC(test_charge, z)
    P = zeros(1,length(z));
    n = length(z);
    for i=1:n
        P(i) = G(test_charge, z(i));
    end
 

  function R=Rcomp(theta,lambda,T,W,Pbc,PWbc,nsub,npan)
  starL=17:80;
  for level=1:nsub
    [z,zp,zpp,nz,w,wzp]=zlocinit(theta,T,W,nsub,level,npan);
    K=MAinit(z,zp,zpp,nz,w,wzp,96);
    %KD = MAinitDL(z,zp,zpp,nz,w,wzp,96);
    %K=KD;
    MAT=eye(96)+lambda*K;
    if level==1
      R=inv(MAT(starL,starL));
    end
    MAT(starL,starL)=inv(R);
    R=PWbc'*(MAT\Pbc);
  end
  
  function R=RcompNew(theta,T,W,Pbc,PWbc,nsub,npan)
  starL=17:80;
  lambda = 1.000;
  for level=1:nsub
    [z,zp,zpp,nz,w,wzp]=zlocinit(theta,T,W,nsub,level,npan);
    K=MAinit(z,zp,zpp,nz,w,wzp,96);
    KD = MAinitDL(z,zp,zpp,nz,w,wzp,96);
    K=KD;
    MAT=eye(96)+lambda*K;
    if level==1
      R=inv(MAT(starL,starL));
    end
    MAT(starL,starL)=inv(R);
    R=PWbc'*(MAT\Pbc);
  end
  
  function [x,it]=myGMRESR(A,R,b,n,m,tol)
% *** GMRES with low-threshold stagnation control ***
  V=zeros(n,m+1);
  H=zeros(m);
  cs=zeros(m,1);
  sn=zeros(m,1);
  bnrm2=norm(b);
  V(:,1)=b/bnrm2;
  s=bnrm2*eye(m+1,1);
  for it=1:m                                  
    it1=it+1;                                   
    w=A*(R*V(:,it));
    for k=1:it
      H(k,it)=V(:,k)'*w;
      w=w-H(k,it)*V(:,k);
    end
    H(it,it)=H(it,it)+1;
    wnrm2=norm(w);
    V(:,it1)=w/wnrm2;
    for k=1:it-1                                
      temp     = cs(k)*H(k,it)+sn(k)*H(k+1,it);
      H(k+1,it)=-sn(k)*H(k,it)+cs(k)*H(k+1,it);
      H(k,it)  = temp;
    end
    [cs(it),sn(it)]=rotmat(H(it,it),wnrm2);     
    H(it,it)= cs(it)*H(it,it)+sn(it)*wnrm2;
    s(it1) =-sn(it)*s(it);                      
    s(it)  = cs(it)*s(it);                         
    myerr=abs(s(it1))/bnrm2;
    if (myerr<=tol)||(it==m)                     
      %disp(['predicted residual = ' num2str(myerr)])
      y=triu(H(1:it,1:it))\s(1:it);             
      x=fliplr(V(:,1:it))*flipud(y);
      trueres=norm(x+A*(R*x)-b)/bnrm2;
      %disp(['true residual      = ',num2str(trueres)])
      break
    end
  end

  function [c,s]=rotmat(a,b)
  if  b==0
    c=1;
    s=0;
  elseif abs(b)>abs(a)
    temp=a/b;
    s=1/sqrt(1+temp^2);
    c=temp*s;
  else
    temp=b/a;
    c=1/sqrt(1+temp^2);
    s=temp*c;
  end

  function M1=MAinit(z,zp,zpp,nz,w,wzp,N)
% *** adjoint of double layer potential ***   
  M1=zeros(N);
  for m=1:N
    M1(:,m)=abs(wzp(m))*real(nz./(z(m)-z));
  end
  M1(1:N+1:N^2)=-w.*imag(zpp./zp)/2;      
  M1=M1/pi;
  
  function M1=MAinitDL(z,zp,zpp,nz,w,wzp,N)
% *** adjoint of double layer potential ***   
  M1=zeros(N);
  for m=1:N
    M1(:,m)=abs(wzp(m))*real(nz(m)./(z-z(m)));
  end
  M1(1:N+1:N^2)=-w.*imag(zpp./zp)/2;      
  M1=-M1/pi;
  
  function M1=MAOffBoundary(z,zp,zpp,nz,w,wzp,N,target)
% *** adjoint of double layer potential ***   
  M1=zeros(1,N);
  for m=1:N
    M1(m)=abs(wzp(m))*real(nz(m)./(target-z(m)));
  end   
  M1=-M1/pi;
  
  function [z,zp,zpp,nz,w,wzp,np]=zinit(theta,sinter,sinterdiff,T,W,npan)
  np=16*npan;
  s=zeros(np,1);
  w=zeros(np,1);
  for k=1:npan
    myind=(k-1)*16+1:k*16;
    sdif=sinterdiff(k)/2;
    s(myind)=(sinter(k)+sinter(k+1))/2+sdif*T;
    w(myind)=W*sdif;
  end
  z=zfunc(s,theta) ;
  zp=zpfunc(s,theta);
  zpp=zppfunc(s,theta);
% *** some extra presicion gained from symmetry ***
  z(np/2+1:np)=conj(flipud(z(1:np/2)));
  zp(np/2+1:np)=-conj(flipud(zp(1:np/2)));
  zpp(np/2+1:np)=conj(flipud(zpp(1:np/2)));
% *************************************************
  nz=-1i*zp./abs(zp);
  wzp=w.*zp;

  function [z,zp,zpp,nz,w,wzp]=zlocinit(theta,T,W,nsub,level,npan)
  denom=2^(nsub-level)*npan;
  s=[T/4+0.25;T/4+0.75;T/2+1.5]/denom;
  w=[W/4;W/4;W/2]/denom;
  w=[flipud(w);w];
  z=zfunc(s,theta);
  z=[conj(flipud(z));z];
  zp=zpfunc(s,theta);
  zp=[-conj(flipud(zp));zp];
  zpp=zppfunc(s,theta);
  zpp=[conj(flipud(zpp));zpp];
  nz=-1i*zp./abs(zp);
  wzp=w.*zp;
  
  function zout=zfunc(s,theta)
  zout=sin(pi*s).*exp(1i*theta*(s-0.5));

  function zpout=zpfunc(s,theta)
  zpout=(pi*cos(pi*s)+1i*theta*sin(pi*s)).*exp(1i*theta*(s-0.5));
  
  function zppout=zppfunc(s,theta)
  zppout=(2i*pi*theta*cos(pi*s)-(theta^2+pi^2)*sin(pi*s)).* ...
	 exp(1i*theta*(s-0.5));  

  function [IP,IPW]=IPinit(T,W)
  A=ones(16);
  AA=ones(32,16);
  T2=[T-1;T+1]/2;
  W2=[W;W]/2;
  for k=2:16
    A(:,k)=A(:,k-1).*T;
    AA(:,k)=AA(:,k-1).*T2;   
  end
  IP=AA/A;
  IPW=IP.*(W2*(1./W)');
  
  function T=Tinit16
% *** 16-point Gauss-Legendre nodes ***  
  T=zeros(16,1);
  T( 1)=-0.989400934991649932596154173450332627;
  T( 2)=-0.944575023073232576077988415534608345;
  T( 3)=-0.865631202387831743880467897712393132;
  T( 4)=-0.755404408355003033895101194847442268;
  T( 5)=-0.617876244402643748446671764048791019;
  T( 6)=-0.458016777657227386342419442983577574;
  T( 7)=-0.281603550779258913230460501460496106;
  T( 8)=-0.095012509837637440185319335424958063;
  T( 9)= 0.095012509837637440185319335424958063;
  T(10)= 0.281603550779258913230460501460496106;
  T(11)= 0.458016777657227386342419442983577574;
  T(12)= 0.617876244402643748446671764048791019;
  T(13)= 0.755404408355003033895101194847442268;
  T(14)= 0.865631202387831743880467897712393132;
  T(15)= 0.944575023073232576077988415534608345;
  T(16)= 0.989400934991649932596154173450332627;

  function W=Winit16
% *** 16-point Gauss-Legendre weights ***  
  W=zeros(16,1); 
  W( 1)= 0.027152459411754094851780572456018104;
  W( 2)= 0.062253523938647892862843836994377694;
  W( 3)= 0.095158511682492784809925107602246226;
  W( 4)= 0.124628971255533872052476282192016420;
  W( 5)= 0.149595988816576732081501730547478549;
  W( 6)= 0.169156519395002538189312079030359962;
  W( 7)= 0.182603415044923588866763667969219939;
  W( 8)= 0.189450610455068496285396723208283105;
  W( 9)= 0.189450610455068496285396723208283105;
  W(10)= 0.182603415044923588866763667969219939;
  W(11)= 0.169156519395002538189312079030359962;
  W(12)= 0.149595988816576732081501730547478549;
  W(13)= 0.124628971255533872052476282192016420;
  W(14)= 0.095158511682492784809925107602246226;
  W(15)= 0.062253523938647892862843836994377694;
  W(16)= 0.027152459411754094851780572456018104;