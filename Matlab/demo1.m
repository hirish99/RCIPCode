  function demo1
% **********************************************
% *** Last changed 2017-03-31 by Johan Helsing *
% *** Solves (4) and computes $q$ of (6)       *
% **********************************************
  close all
  format long
  format compact
% *** User specified quantities ***************
  nsub=40;       % number of subdivisions
  lambda=0.999;  % parameter lambda
  theta=pi/2;    % corner opening angle
  npan=10;       % number of coarse panels
  evec=1;        % a unit vector 
  qref=1.1300163213105365; % reference solution
% *********************************************
  T=Tinit16;
  W=Winit16;
%
% *** Panels, discretization points, and weights ***  
  [sinterfin,sinterfindiff,npanfin]=panelinit(nsub,npan);
  [zfin,zpfin,zppfin,nzfin,wfin,wzpfin,npfin]=zinit(theta,sinterfin, ...
					sinterfindiff,T,W,npanfin);
  arclength=sum(abs(wzpfin));
  disp(['arclength = ',num2str(arclength,16)])
  disp(['area      = ',num2str(0.5*imag(conj(zfin).'*wzpfin),15)])
%
% *** The K_fin matrix is set up ***
  Kfin=MAinit(zfin,zpfin,zppfin,nzfin,wfin,wzpfin,npfin);
  test=ones(npfin,1);
  disp(['check Kfin = ',num2str(abs(wzpfin)'*(Kfin*test)+arclength)])
%
% *** Solving main linear system ***
  rhs=2*lambda*real(conj(evec)*nzfin);     
  [rhofin,it]=myGMRES(lambda*Kfin,rhs,npfin,100,eps);
  disp(['GMRES iter = ',num2str(it)])
%
% *** Post processing ***
  qfin=rhofin.'*real(conj(evec)*zfin.*abs(wzpfin))
  disp(['estimated error = ',num2str(abs(qref-qfin)/abs(qref))])
  
  function [x,it]=myGMRES(A,b,n,m,tol)
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
    w=A*V(:,it);
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
      disp(['predicted residual = ' num2str(myerr)])
      y=triu(H(1:it,1:it))\s(1:it);             
      x=fliplr(V(:,1:it))*flipud(y);
      trueres=norm(x+A*x-b)/bnrm2;
      disp(['true residual      = ',num2str(trueres)])
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

  function [sinter,sinterdiff,npanfin]=panelinit(nsub,npan)
  npanfin=npan+2*nsub;
  sinter=zeros(npanfin+1,1);
  sinter(1:npan+1)=linspace(0,1,npan+1);
  sinterdiff=ones(npan+2*nsub,1)/npan;
  for k=1:nsub
    sinter(3:end)=sinter(2:end-1);
    sinter(2)=(sinter(1)+sinter(2))/2;   
    sinterdiff(3:end)=sinterdiff(2:end-1);
    sinterdiff(2)=sinterdiff(1)/2;
    sinterdiff(1)=sinterdiff(1)/2;   
  end
  sinter(end-nsub:end)=1-flipud(sinter(1:nsub+1));
  sinterdiff(end-nsub-1:end)=flipud(sinterdiff(1:nsub+2));
  
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
  
  function zout=zfunc(s,theta)
  zout=sin(pi*s).*exp(1i*theta*(s-0.5));

  function zpout=zpfunc(s,theta)
  zpout=(pi*cos(pi*s)+1i*theta*sin(pi*s)).*exp(1i*theta*(s-0.5));
  
  function zppout=zppfunc(s,theta)
  zppout=(2i*pi*theta*cos(pi*s)-(theta^2+pi^2)*sin(pi*s)).* ...
	 exp(1i*theta*(s-0.5));  

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