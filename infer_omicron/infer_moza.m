function infer_moza()
%%inference without delay

close all

load Pop
load mozadaily


% %%%%%%%%%%%%%%%
%ts=1; %tintegration start time
num_ens=300;
num_times=size(mozadaily,1);
obs_truth=mozadaily';
lambda=1.015;%inflation factor
num_loc=size(Pop,1);%number of locations
N=Pop; %population
% %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%#ADD EAKF
%%%%set OEV which is required by EAKF. OEV is used by the EAKF to weigh the observations
%%%%# and model prior estimation of covid incidence, and to produce the model posterior estitmation 
% %%%#of both the state variables and parameters

OEV=zeros(num_loc,num_times);
for l=1:num_loc
    for t=1:num_times
          OEV(l,t)=max(4,obs_truth(l,t)^2/4); 
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%# start EAKF of network model
%%%%%initialization

x=initialize_eakf(Pop,num_ens);
x =checkbound(x,Pop);


%%%%run for one day to initialize system

for t=580:581 %1 day
    [x,obs_ens]=model_eakf(x,t,Pop);
end  

%Begin looping through observations
num_var=size(x,1); %number of state variables
xprior=NaN(num_var,num_ens,num_times);%prior
xpost=xprior;


 
 %for tt=101:num_times
 for tt=581:728

    %%% inflation of x before assimilation to avoid ensemble collapse
    x([1:55,59:80],:)=mean(x([1:55,59:80],:),2)*ones(1,num_ens)+lambda*(x([1:55,59:80],:)-mean(x([1:55,59:80],:),2)*ones(1,num_ens));

    x =checkbound(x,Pop);

    %%%  Integrate forward one time step
        [x,obs_ens]=model_eakf(x,tt,Pop);
           

    
    xprior(:,:,tt)=x;%set prior
   
            
     %%loop through local observations

   for l=1:num_loc
            %Get the variance of the ensemble
            obs_var = OEV(l,tt);
            prior_var = var(obs_ens(l,:));
            post_var = prior_var*obs_var/(prior_var+obs_var);
                if prior_var==0%if degenerate
                    post_var=1e-3;
                    prior_var=1e-3;
                end
            prior_mean = mean(obs_ens(l,:));
            post_mean = post_var*(prior_mean/prior_var + obs_truth(l,tt)/obs_var);


%%%% Compute alphaa and adjust distribution to conform to posterior moments
        alpha = (obs_var/(obs_var+prior_var)).^0.5;
        dy = post_mean + alpha*(obs_ens(l,:)-prior_mean)-obs_ens(l,:);%%#observation adjustment



       %%%%%%%%%%%%%%%%%%%%%%%%%%%%Loop over each state variable
       rr=zeros(1,num_var);
        %%% SOME INDEX OF THE OBSERVATION/PROVINCE NUMBER jz 
    for idx=l
        for jz=1:5
            A=cov(x(5*(idx-1)+jz,:),obs_ens(l,:)); %%# covariance btwn each unobserved variabl/param nd observations calculated 4rm ensemble
            rr(5*(idx-1)+jz)=A(2,1)/prior_var;
        end
    end

        
          
       %%%%ADJUST PARAMETERS GLOBALLY except \alpha & beta 56-58
    
            for jw=num_loc*5+1:num_loc*5+3 %%%56:58 mu,Z,D
                A=cov(x(jw,:),obs_ens(l,:)); %%# covariance btwn each unobserved variabl/param nd observations calculated 4rm ensemble
                rr(jw)=A(2,1)/prior_var;
            end


            %%%%ADJUST \BETA LOCALLY  59-69

            for idx=l
                A=cov(x(num_loc*5+3+idx,:),obs_ens(l,:));   %%# covariance btwn each unobserved variabl/param nd observations calculated 4rm ensemble
                rr(num_loc*5+3+idx)=A(2,1)/prior_var;
            end


        %%%%ADJUST \ALPHA LOCALLY 70-80

            for idx=l
                A=cov(x(num_loc*5+14+idx,:),obs_ens(l,:));   %%# covariance btwn each unobserved variabl/param nd observations calculated 4rm ensemble
                rr(num_loc*5+14+idx)=A(2,1)/prior_var;
            end


    % end
        %%Get the adjusted variable 
        %%%# adjustments of unobserved variables/params r determined
        %%%bymultiplying the covariance rr with observation adjustment dy
        dx=rr'*dy;

        vardx(:,:,l)=dx(1:55,:);
        pramGLOBAL_dx(:,:,l)=dx(56:58,:);
        pramLOCAL_beta_dx(:,:,l)=dx(59:69,:);
        pramLOCAL_alpha_dx(:,:,l)=dx(70:end,:);


        sum_vardx=sum(vardx,3);
        sum_pramLOCAL_beta_dx =sum(pramLOCAL_beta_dx,3);
        sum_pramLOCAL_alpha_dx =sum(pramLOCAL_alpha_dx,3);
        sum_pramGLOBAL_dx =sum(pramGLOBAL_dx,3);

        avg_pramGLOBAL_dx=sum_pramGLOBAL_dx/num_loc;

        dx2 = [sum_vardx;avg_pramGLOBAL_dx;sum_pramLOCAL_beta_dx;sum_pramLOCAL_alpha_dx];
               
        obs_ens(l,:)=obs_ens(l,:)+dy;
        obs_ens(l,obs_ens(l,:)<0)=0;
    
   end

         x = x + dx2;
        %Corrections to DA produced aphysicalities
        x = checkbound(x,Pop);
        xnew = x;
        xpost(:,:,tt)=xnew;
        
       
 end

%%%%# Plots



%  %%%  Get the ensemble mean of the prior and posterior at each time step
um=shiftdim(xprior,1);

um=shiftdim(xpost,1);
xpost_mean=squeeze(mean(um));


% %%%%%%%%%%%%%%%#PLOT parameters/r0 CI
tm=datetime(2020,3,30)+days(581:728);


%%%%SIMULATED Rt
%%%cabo delgabo
r01=xpost(59,:,581:728).*xpost(58,:,581:728).*(xpost(70,:,581:728)+(1-xpost(70,:,581:728)).*xpost(56,:,581:728));
um=shiftdim(r01,1);
r01_mean=squeeze(mean(um));

%%%gaza
r02=xpost(60,:,581:728).*xpost(58,:,581:728).*(xpost(71,:,581:728)+(1-xpost(71,:,581:728)).*xpost(56,:,581:728));
um=shiftdim(r02,1);
r02_mean=squeeze(mean(um));

%%%inhambane
r03=xpost(61,:,581:728).*xpost(58,:,581:728).*(xpost(72,:,581:728)+(1-xpost(72,:,581:728)).*xpost(56,:,581:728));
um=shiftdim(r03,1);
r03_mean=squeeze(mean(um));

%%%manica
r04=xpost(62,:,581:728).*xpost(58,:,581:728).*(xpost(73,:,581:728)+(1-xpost(73,:,581:728)).*xpost(56,:,581:728));
um=shiftdim(r04,1);
r04_mean=squeeze(mean(um));

%%%maputo cidade
r05=xpost(63,:,581:728).*xpost(58,:,581:728).*(xpost(74,:,581:728)+(1-xpost(74,:,581:728)).*xpost(56,:,581:728));
um=shiftdim(r05,1);
r05_mean=squeeze(mean(um));

%%%maputo provincia
r06=xpost(64,:,581:728).*xpost(58,:,581:728).*(xpost(75,:,581:728)+(1-xpost(75,:,581:728)).*xpost(56,:,581:728));
um=shiftdim(r06,1);
r06_mean=squeeze(mean(um));

%%%nampula
r07=xpost(65,:,581:728).*xpost(58,:,581:728).*(xpost(76,:,581:728)+(1-xpost(76,:,581:728)).*xpost(56,:,581:728));
um=shiftdim(r07,1);
r07_mean=squeeze(mean(um));

%%%niassa
r08=xpost(66,:,581:728).*xpost(58,:,581:728).*(xpost(77,:,581:728)+(1-xpost(77,:,581:728)).*xpost(56,:,581:728));
um=shiftdim(r08,1);
r08_mean=squeeze(mean(um));

%%%sofala
r09=xpost(67,:,581:728).*xpost(58,:,581:728).*(xpost(78,:,581:728)+(1-xpost(78,:,581:728)).*xpost(56,:,581:728));
um=shiftdim(r09,1);
r09_mean=squeeze(mean(um));

%%%tete
r010=xpost(68,:,581:728).*xpost(58,:,581:728).*(xpost(79,:,581:728)+(1-xpost(79,:,581:728)).*xpost(56,:,581:728));
um=shiftdim(r010,1);
r010_mean=squeeze(mean(um));

%%%zambezia
r011=xpost(69,:,581:728).*xpost(58,:,581:728).*(xpost(80,:,581:728)+(1-xpost(80,:,581:728)).*xpost(56,:,581:728));
um=shiftdim(r011,1);
r011_mean=squeeze(mean(um));



%%%%%%%



       %%percentiles cabo
Pr01=r01(1,:,:);
Pr01_low=prctile(Pr01,2.5,2);
Pr01_high=prctile(Pr01,97.5,2);

%%gaza
Pr02=r02(1,:,:);
Pr02_low=prctile(Pr02,2.5,2);
Pr02_high=prctile(Pr02,97.5,2);

%%inhambane
Pr03=r03(1,:,:);
Pr03_low=prctile(Pr03,2.5,2);
Pr03_high=prctile(Pr03,97.5,2);


%%manica
Pr04=r04(1,:,:);
Pr04_low=prctile(Pr04,2.5,2);
Pr04_high=prctile(Pr04,97.5,2);


%%maputo cidade
Pr05=r05(1,:,:);
Pr05_low=prctile(Pr05,2.5,2);
Pr05_high=prctile(Pr05,97.5,2);


%%maputo
Pr06=r06(1,:,:);
Pr06_low=prctile(Pr06,2.5,2);
Pr06_high=prctile(Pr06,97.5,2);


%%nampula
Pr07=r07(1,:,:);
Pr07_low=prctile(Pr07,2.5,2);
Pr07_high=prctile(Pr07,97.5,2);


%%niassa
Pr08=r08(1,:,:);
Pr08_low=prctile(Pr08,2.5,2);
Pr08_high=prctile(Pr08,97.5,2);


%%sofala
Pr09=r09(1,:,:);
Pr09_low=prctile(Pr09,2.5,2);
Pr09_high=prctile(Pr09,97.5,2);


%%tete
Pr010=r010(1,:,:);
Pr010_low=prctile(Pr010,2.5,2);
Pr010_high=prctile(Pr010,97.5,2);


%%zambezia
Pr011=r011(1,:,:);
Pr011_low=prctile(Pr011,2.5,2);
Pr011_high=prctile(Pr011,97.5,2);



% % 
 %%%%%%%%%parameters and ensemble    
%%%%%%%%%%%%%%%%ALL
Ppara_all=xpost(:,:,581:728);
%Ppara_all=xpost(:,:,:);
Ppara_low=prctile(Ppara_all,2.5,2);
Ppara_high=prctile(Ppara_all,97.5,2);


xx=tm;

%%%%RE percentiles
%%%cabo
yy1=squeeze(Pr01_low(1,1,:))';
yy2=squeeze(Pr01_high(1,1,:))';
%%%gaza
yy1002=squeeze(Pr02_low(1,1,:))';
yy2002=squeeze(Pr02_high(1,1,:))';

%%%inhambane
yy1003=squeeze(Pr03_low(1,1,:))';
yy2003=squeeze(Pr03_high(1,1,:))';

%%%manica
yy1004=squeeze(Pr04_low(1,1,:))';
yy2004=squeeze(Pr04_high(1,1,:))';

%%%maputo cidade
yy1005=squeeze(Pr05_low(1,1,:))';
yy2005=squeeze(Pr05_high(1,1,:))';

%%%maputo provincia
yy1006=squeeze(Pr06_low(1,1,:))';
yy2006=squeeze(Pr06_high(1,1,:))';

%%%nampula
yy1007=squeeze(Pr07_low(1,1,:))';
yy2007=squeeze(Pr07_high(1,1,:))';

%%%niassa
yy1008=squeeze(Pr08_low(1,1,:))';
yy2008=squeeze(Pr08_high(1,1,:))';

%%%sofala
yy1009=squeeze(Pr09_low(1,1,:))';
yy2009=squeeze(Pr09_high(1,1,:))';

%%%tete
yy10010=squeeze(Pr010_low(1,1,:))';
yy20010=squeeze(Pr010_high(1,1,:))';

%%%zambezia
yy10011=squeeze(Pr011_low(1,1,:))';
yy20011=squeeze(Pr011_high(1,1,:))';



%%%%%beta pcentiles
yy12=squeeze(Ppara_low(59,1,:))';
yy22=squeeze(Ppara_high(59,1,:))';

yy13=squeeze(Ppara_low(60,1,:))';
yy23=squeeze(Ppara_high(60,1,:))';

yy14=squeeze(Ppara_low(61,1,:))';
yy24=squeeze(Ppara_high(61,1,:))';

yy15=squeeze(Ppara_low(62,1,:))';
yy25=squeeze(Ppara_high(62,1,:))';

yy16=squeeze(Ppara_low(63,1,:))';
yy26=squeeze(Ppara_high(63,1,:))';

yy17=squeeze(Ppara_low(64,1,:))';
yy27=squeeze(Ppara_high(64,1,:))';

yy18=squeeze(Ppara_low(65,1,:))';
yy28=squeeze(Ppara_high(65,1,:))';

yy19=squeeze(Ppara_low(66,1,:))';
yy29=squeeze(Ppara_high(66,1,:))';

yy110=squeeze(Ppara_low(67,1,:))';
yy210=squeeze(Ppara_high(67,1,:))';

yy111=squeeze(Ppara_low(68,1,:))';
yy211=squeeze(Ppara_high(68,1,:))';

yy112=squeeze(Ppara_low(69,1,:))';
yy212=squeeze(Ppara_high(69,1,:))';

%%%%%%Alpha percentiles
%%
yy113=squeeze(Ppara_low(70,1,:))';
yy213=squeeze(Ppara_high(70,1,:))';

yy114=squeeze(Ppara_low(71,1,:))';
yy214=squeeze(Ppara_high(71,1,:))';


yy115=squeeze(Ppara_low(72,1,:))';
yy215=squeeze(Ppara_high(72,1,:))';


yy116=squeeze(Ppara_low(73,1,:))';
yy216=squeeze(Ppara_high(73,1,:))';


yy117=squeeze(Ppara_low(74,1,:))';
yy217=squeeze(Ppara_high(74,1,:))';


yy118=squeeze(Ppara_low(75,1,:))';
yy218=squeeze(Ppara_high(75,1,:))';


yy119=squeeze(Ppara_low(76,1,:))';
yy219=squeeze(Ppara_high(76,1,:))';


yy1120=squeeze(Ppara_low(77,1,:))';
yy2120=squeeze(Ppara_high(77,1,:))';


yy1121=squeeze(Ppara_low(78,1,:))';
yy2121=squeeze(Ppara_high(78,1,:))';


yy1122=squeeze(Ppara_low(79,1,:))';
yy2122=squeeze(Ppara_high(79,1,:))';

yy1123=squeeze(Ppara_low(80,1,:))';
yy2123=squeeze(Ppara_high(80,1,:))';


%%%%%parameter time series


figure(31)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy12, fliplr(yy22)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,59),'r','Linewidth',2)% location 1 
        title('\beta_1 Cabo Delgado')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy113, fliplr(yy213)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,70),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_1')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1, fliplr(yy2)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r01_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(32)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy13, fliplr(yy23)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,60),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_2 Gaza')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy114, fliplr(yy214)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,71),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_2')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1002, fliplr(yy2002)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r02_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(33)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy14, fliplr(yy24)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,61),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_3 Inhambane')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy115, fliplr(yy215)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,72),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_3')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1003, fliplr(yy2003)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r03_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(34)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy15, fliplr(yy25)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,62),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_4 Manica')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy116, fliplr(yy216)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,73),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_4')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1004, fliplr(yy2004)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r04_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(35)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy16, fliplr(yy26)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,63),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_5 Maputo Cidade')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy117, fliplr(yy217)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,74),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_5')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1005, fliplr(yy2005)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r05_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(36)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy17, fliplr(yy27)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,64),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_6 Maputo')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy118, fliplr(yy218)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,75),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_6')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1006, fliplr(yy2006)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r06_mean(:,:),'r','Linewidth',2)% location 1 
        %yline(2.5664,'k--','Linewidth',1.5);
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(37)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy18, fliplr(yy28)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,65),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_7 Nampula')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy119, fliplr(yy219)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,76),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_7')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1007, fliplr(yy2007)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r07_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(38)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy19, fliplr(yy29)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,66),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_8 Niassa')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1120, fliplr(yy2120)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,77),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_8')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1008, fliplr(yy2008)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r08_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(39)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy110, fliplr(yy210)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,67),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_9 Sofala')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1121, fliplr(yy2121)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,78),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_9')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1009, fliplr(yy2009)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r09_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(40)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy111, fliplr(yy211)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,68),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_{10} Tete')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1122, fliplr(yy2122)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,79),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_{10}')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy10010, fliplr(yy20010)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r010_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)


        figure(41)
        subplot(3,1,1)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy112, fliplr(yy212)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,69),'r','Linewidth',2)% location 1 
        ylabel('\beta prior', 'FontSize', 11)
        title('\beta_{11} Zambezia')
        legend('95% CrI ','posterior mean','Location','Best')
        legend boxoff 
        set (gca,'FontSize',12)

        subplot(3,1,2)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy1123, fliplr(yy2123)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,xpost_mean(581:728,80),'r','Linewidth',2)% location 1 
        ylabel('\alpha prior ', 'FontSize', 11)
        title('\alpha_{11}')
        set (gca,'FontSize',12)
        
        subplot(3,1,3)
        x2 = [xx, fliplr(xx)];
        inBetween = [yy10011, fliplr(yy20011)];
        fill(x2, inBetween, 'g','FaceAlpha',0.3,'EdgeColor', 'none');
        hold on
        plot(tm,r011_mean(:,:),'r','Linewidth',2)% location 1 
        xlabel('Time (days)', 'FontSize', 10)
        ylabel('R_t', 'FontSize', 11)
        set (gca,'FontSize',12)
%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%        


%%%%observations

yy31=squeeze(Ppara_low(5,1,:))';
yy41=squeeze(Ppara_high(5,1,:))';

yy321=squeeze(Ppara_low(10,1,:))';
yy421=squeeze(Ppara_high(10,1,:))';

yy331=squeeze(Ppara_low(15,1,:))';
yy431=squeeze(Ppara_high(15,1,:))';

yy341=squeeze(Ppara_low(20,1,:))';
yy441=squeeze(Ppara_high(20,1,:))';

yy351=squeeze(Ppara_low(25,1,:))';
yy451=squeeze(Ppara_high(25,1,:))';

yy361=squeeze(Ppara_low(30,1,:))';
yy461=squeeze(Ppara_high(30,1,:))';

yy371=squeeze(Ppara_low(35,1,:))';
yy471=squeeze(Ppara_high(35,1,:))';

yy381=squeeze(Ppara_low(40,1,:))';
yy481=squeeze(Ppara_high(40,1,:))';

yy391=squeeze(Ppara_low(45,1,:))';
yy491=squeeze(Ppara_high(45,1,:))';

yy301=squeeze(Ppara_low(50,1,:))';
yy401=squeeze(Ppara_high(50,1,:))';

yy3001=squeeze(Ppara_low(55,1,:))';
yy4001=squeeze(Ppara_high(55,1,:))';

data=obs_truth(:,581:728);

%%%%%%%%%%%%%plots

figure (91)

 x2 = [xx, fliplr(xx)];
inBetween = [yy31, fliplr(yy41)];
fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
hold on
plot(tm,xpost_mean(581:728,5),'r-','Linewidth', 1.5)
hold on
plot(tm, data(1,:), 'b.')
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Cabo Delgado ','FontWeight','bold')
legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)

 figure (92)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy321, fliplr(yy421)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(581:728,10),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(2,:), 'b.')
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Gaza ','FontWeight','bold')
legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)

 figure (93)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy331, fliplr(yy431)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(581:728,15),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(3,:), 'b.')
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Inhambane ','FontWeight','bold')
legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16) 

 figure (94)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy341, fliplr(yy441)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(581:728,20),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(4,:), 'b.')
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Manica ','FontWeight','bold')
legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)


figure (95)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy351, fliplr(yy451)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(581:728,25),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(5,:), 'b.')
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Maputo Cidade ','FontWeight','bold')
legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16) 

figure (96)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy361, fliplr(yy461)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(581:728,30),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(6,:), 'b.')
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Maputo ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16) 


 figure (97)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy371, fliplr(yy471)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(581:728,35),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(7,:), 'b.')
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Nampula ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)

 figure (98)

 x2 = [xx, fliplr(xx)];
 inBetween = [yy381, fliplr(yy481)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(581:728,40),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(8,:), 'b.')
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Niassa ','FontWeight','bold')
legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)

 figure (99)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy391, fliplr(yy491)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(581:728,45),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(9,:), 'b.')
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Sofala ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)

 figure (100)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy301, fliplr(yy401)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(581:728,50),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(10,:), 'b.')
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Tete ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)

 figure (101)
 x2 = [xx, fliplr(xx)];
 inBetween = [yy3001, fliplr(yy4001)];
 fill(x2, inBetween, 'r','FaceAlpha',0.3,'EdgeColor', 'none');
 hold on
 plot(tm,xpost_mean(581:728,55),'r-','Linewidth', 1.5)
 hold on
 plot(tm, data(11,:), 'b.')
 xlabel('Date', 'FontSize', 12.5)
 ylabel('Daily new cases (population)', 'FontSize', 12.5)
 title('Zambezia ','FontWeight','bold')
 legend('95% CrI','New cases posterior mean','Data', 'FontSize', 10,'Location','Best')
 legend boxoff 
 set (gca,'FontSize',16)


 


 figure()


%%%%%%%%%%%%%%%#CHECKBOUND inference


function x=checkbound(x,Pop)

num_loc=size(Pop,1);
N=Pop;
 %%%parameters
betal=0.5;betau=2;%transmission rate 
mul=0.8;muu=0.8;%relative transmissibility
Zl=4; Zu=4;
alphal=0.004;alphau=0.1;
Dl=4; Du=4; %infectious period 
% 
xmin=[mul,Zl,Dl,ones(1,num_loc)*betal,ones(1,num_loc)*alphal];
xmax=[muu,Zu,Du,ones(1,num_loc)*betau,ones(1,num_loc)*alphau];


%%%lower bounds \alpha
xmin(15)=0.007;
xmin(16)=0.02;
xmin(17)=0.03;
xmin(18)=0.01;
xmin(19)=0.14;
xmin(20)=0.03;
xmin(21)=0.004;
xmin(22)=0.01;
xmin(23)=0.01;
xmin(24)=0.009;
xmin(25)=0.005;


xmax(19)=0.2;%maputo cidade



%%%redistribute out of bound parmaeter members
for i=1:25
    temp=x(end-25+i,:);
    index=(temp<xmin(i))|(temp>xmax(i));
    index_out=find(index>0);
    index_in=find(index==0);
    %redistribute out bound ensemble members
    x(end-25+i,index_out)=datasample(x(end-25+i,index_in),length(index_out));
end







%%state variables
for i=1:num_loc

     %S
  
    x((i-1)*5+1,x((i-1)*5+1,:)>N(i))=N(i)-1;

    %     %E
       x((i-1)*5+2,x((i-1)*5+2,:)>N(i))=N(i)-1;
    %Ir
        x((i-1)*5+3,x((i-1)*5+3,:)>N(i))=N(i)-1;
    %Iu
       x((i-1)*5+4,x((i-1)*5+4,:)>N(i))=N(i)-1;
    %obs
        x((i-1)*5+5,x((i-1)*5+5,:)>N(i))=N(i)-1;




end
% 
% %%%%redistribute out of bound state varaibles members
for i=1:num_loc
    for jj=1:5
    temp_var=x((i-1)*5+jj,:);
        index_var=(temp_var<0);
        index_var_out=find(index_var>0);
        index_var_in=find(index_var==0);
        %redistribute out bound ensemble members
    x((i-1)*5+jj,index_var_out)=datasample(x((i-1)*5+jj,index_var_in),length(index_var_out));
    end
end











