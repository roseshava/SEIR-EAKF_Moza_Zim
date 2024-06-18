function [x,obs]=model_eakf(x,ts,Pop)

dt=1; %integrate forward for one day
tmstep=1;%data is daily
[~,num_ens]=size(x);
num_loc=size(Pop,1);
N=Pop; %population

%S,E,Is,Ia,obs,...,beta,mu,theta,Z,alpha,D
Sidx=(1:5:5*num_loc)';
Eidx=(2:5:5*num_loc)';
Iridx=(3:5:5*num_loc)';
Iuidx=(4:5:5*num_loc)';
obsidx=(5:5:5*num_loc)';
%%%%%%%%%%%%%%%%%%%%%%%%%%
muidx=num_loc*5+1; %mu: adjustable parameter
Zidx=num_loc*5+2; %alpha: adjustable parameter
Didx=num_loc*5+3;  %D: duration of infection

betaidx= (num_loc*5+4:num_loc*5+14)';
alphaidx= (num_loc*5+15:num_loc*5+25)';



%%transform format from state space vector to matrix
S=zeros(num_loc,num_ens,abs(tmstep)+1);
E=zeros(num_loc,num_ens,abs(tmstep)+1);
Ir=zeros(num_loc,num_ens,abs(tmstep)+1);
Iu=zeros(num_loc,num_ens,abs(tmstep)+1);

Incidence=zeros(num_loc,num_ens,abs(tmstep)+1);
obs=zeros(num_loc,num_ens);



%%%%%%%%%%%%%%%%%%%%%%% 

%initialize S,E,Ir,Iu and parameters
S(:,:,1)=x(Sidx,:);
E(:,:,1)=x(Eidx,:);
Ir(:,:,1)=x(Iridx,:);
Iu(:,:,1)=x(Iuidx,:);

mu=x(muidx,:);
Z=x(Zidx,:);
D=x(Didx,:);
beta=x(betaidx,:);
alpha=x(alphaidx,:);


%%%%%%%%%%%%%%%%%%%%%
%start integration

tcnt=0;

for t=ts+dt:dt:ts+tmstep
    tcnt=tcnt+1; %Run model forward


    %first step
   
    dt1=dt;

    Eexpr=dt1*(ones(num_loc,1).*beta).*S(:,:,tcnt).*Ir(:,:,tcnt)./N;
    Eexpu=dt1*(ones(num_loc,1)*mu).*(ones(num_loc,1).*beta).*S(:,:,tcnt).*Iu(:,:,tcnt)./N;
    Einfr=dt1*(ones(num_loc,1).*alpha).*E(:,:,tcnt)./(ones(num_loc,1)*Z);
    Einfu=dt1*(ones(num_loc,1).*(1-alpha)).*E(:,:,tcnt)./(ones(num_loc,1)*Z);
    Erecr=dt1*Ir(:,:,tcnt)./(ones(num_loc,1)*D);
    Erecu=dt1*Iu(:,:,tcnt)./(ones(num_loc,1)*D);
    
   
    Eexpr=max(Eexpr,0);Eexpu=max(Eexpu,0);
    Einfr=max(Einfr,0);Einfu=max(Einfu,0);
    Erecr=max(Erecr,0);Erecu=max(Erecu,0);

    %%%%%%STOCHASTIC version
    Eexpr=poissrnd(Eexpr);
    Eexpu=poissrnd(Eexpu);
    Einfr=poissrnd(Einfr);
    Einfu=poissrnd(Einfu);
    Erecr=poissrnd(Erecr);
    Erecu=poissrnd(Erecu);

        
    sk1=-Eexpr-Eexpu;
    ek1=Eexpr+Eexpu-Einfr-Einfu;
    irk1=Einfr-Erecr;
    iuk1=Einfu-Erecu;
    ik1i=Einfr;

          
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%second step


    Ts1=S(:,:,tcnt)+sk1/2;
    Te1=E(:,:,tcnt)+ek1/2;
    Tir1=Ir(:,:,tcnt)+irk1/2;
    Tiu1=Iu(:,:,tcnt)+iuk1/2;


    Eexpr=dt1*(ones(num_loc,1).*beta).*Ts1.*Tir1./N;
    Eexpu=dt1*(ones(num_loc,1)*mu).*(ones(num_loc,1).*beta).*Ts1.*Tiu1./N;
    Einfr=dt1*(ones(num_loc,1).*alpha).*Te1./(ones(num_loc,1)*Z);
    Einfu=dt1*(ones(num_loc,1).*(1-alpha)).*Te1./(ones(num_loc,1)*Z);
    Erecr=dt1*Tir1./(ones(num_loc,1)*D);
    Erecu=dt1*Tiu1./(ones(num_loc,1)*D);

    
   
    Eexpr=max(Eexpr,0);Eexpu=max(Eexpu,0);
    Einfr=max(Einfr,0);Einfu=max(Einfu,0);
    Erecr=max(Erecr,0);Erecu=max(Erecu,0);

     %%%%%%STOCHASTIC version
    Eexpr=poissrnd(Eexpr);
    Eexpu=poissrnd(Eexpu);
    Einfr=poissrnd(Einfr);
    Einfu=poissrnd(Einfu);
    Erecr=poissrnd(Erecr);
    Erecu=poissrnd(Erecu);
   
    sk2=-Eexpr-Eexpu;
    ek2=Eexpr+Eexpu-Einfr-Einfu;
    irk2=Einfr-Erecr;
    iuk2=Einfu-Erecu;
    ik2i=Einfr;

   

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %third step
    Ts2=S(:,:,tcnt)+sk2/2;
    Te2=E(:,:,tcnt)+ek2/2;
    Tir2=Ir(:,:,tcnt)+irk2/2;
    Tiu2=Iu(:,:,tcnt)+iuk2/2;


    Eexpr=dt1*(ones(num_loc,1).*beta).*Ts2.*Tir2./N;
    Eexpu=dt1*(ones(num_loc,1)*mu).*(ones(num_loc,1).*beta).*Ts2.*Tiu2./N;
    Einfr=dt1*(ones(num_loc,1).*alpha).*Te2./(ones(num_loc,1)*Z);
    Einfu=dt1*(ones(num_loc,1).*(1-alpha)).*Te2./(ones(num_loc,1)*Z);
    Erecr=dt1*Tir2./(ones(num_loc,1)*D);
    Erecu=dt1*Tiu2./(ones(num_loc,1)*D);

   
    Eexpr=max(Eexpr,0);Eexpu=max(Eexpu,0);
    Einfr=max(Einfr,0);Einfu=max(Einfu,0);
    Erecr=max(Erecr,0);Erecu=max(Erecu,0);

     %%%%%%STOCHASTIC version
    Eexpr=poissrnd(Eexpr);
    Eexpu=poissrnd(Eexpu);
    Einfr=poissrnd(Einfr);
    Einfu=poissrnd(Einfu);
    Erecr=poissrnd(Erecr);
    Erecu=poissrnd(Erecu);
    
   
    sk3=-Eexpr-Eexpu;
    ek3=Eexpr+Eexpu-Einfr-Einfu;
    irk3=Einfr-Erecr;
    iuk3=Einfu-Erecu;
    ik3i=Einfr;

   
    %fourth step
 
    Ts3=S(:,:,tcnt)+sk3;
    Te3=E(:,:,tcnt)+ek3;
    Tir3=Ir(:,:,tcnt)+irk3;
    Tiu3=Iu(:,:,tcnt)+iuk3; 
    
    Eexpr=dt1*(ones(num_loc,1).*beta).*Ts3.*Tir3./N;
    Eexpu=dt1*(ones(num_loc,1)*mu).*(ones(num_loc,1).*beta).*Ts3.*Tiu3./N;
    Einfr=dt1*(ones(num_loc,1).*alpha).*Te3./(ones(num_loc,1)*Z);
    Einfu=dt1*(ones(num_loc,1).*(1-alpha)).*Te3./(ones(num_loc,1)*Z);
    Erecr=dt1*Tir3./(ones(num_loc,1)*D);
    Erecu=dt1*Tiu3./(ones(num_loc,1)*D);

    

    Eexpr=max(Eexpr,0);Eexpu=max(Eexpu,0);
    Einfr=max(Einfr,0);Einfu=max(Einfu,0);
    Erecr=max(Erecr,0);Erecu=max(Erecu,0);

    %%%%%%STOCHASTIC version
    Eexpr=poissrnd(Eexpr);
    Eexpu=poissrnd(Eexpu);
    Einfr=poissrnd(Einfr);
    Einfu=poissrnd(Einfu);
    Erecr=poissrnd(Erecr);
    Erecu=poissrnd(Erecu);
    
    sk4=-Eexpr-Eexpu;
    ek4=Eexpr+Eexpu-Einfr-Einfu;
    irk4=Einfr-Erecr;
    iuk4=Einfu-Erecu;
    ik4i=Einfr;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    S(:,:,tcnt+1)=S(:,:,tcnt)+round(sk1/6+sk2/3+sk3/3+sk4/6);
    E(:,:,tcnt+1)=E(:,:,tcnt)+round(ek1/6+ek2/3+ek3/3+ek4/6);
    Ir(:,:,tcnt+1)=Ir(:,:,tcnt)+round(irk1/6+irk2/3+irk3/3+irk4/6);
    Iu(:,:,tcnt+1)=Iu(:,:,tcnt)+round(iuk1/6+iuk2/3+iuk3/3+iuk4/6);
    Incidence(:,:,tcnt+1)=round(ik1i/6+ik2i/3+ik3i/3+ik4i/6);
    obs=Incidence(:,:,tcnt+1);

    
end


%%%%%%%%%%%%%%%%%
%update x

%%%update x
x(Sidx,:)=S(:,:,tcnt+1);
x(Eidx,:)=E(:,:,tcnt+1);
x(Iridx,:)=Ir(:,:,tcnt+1);
x(Iuidx,:)=Iu(:,:,tcnt+1);
x(obsidx,:)=obs;





