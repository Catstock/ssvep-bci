function new_Q_table=Q_sd_train_f(episode,tw,mw,threshold,partition,alpha,gamma,train_time,fs,stimulus,eeg,eeglabel,Q_table)
warning off;
EPISODES=episode;  %iteration times
REWARD=70;    
ALPHA=alpha;    %learning rate
GAMMA=gamma;    %attenuation rate
STATES=7777;    %0:0.5:3

[channel,frame,trail]=size(eeg);
eeg=permute(eeg,[2,1,3]);   %frame*channel*trail
[~,choice]=size(stimulus);
window_num=floor((train_time-tw)/mw+1);


for episode=1:EPISODES
    EPSILION=episode/EPISODES; %choose strategy
    for i=1:trail
        Smf=ones(1,choice);   %initialize
        Prf=zeros(1,choice);
        Pi=zeros(1,choice);
        window_start_time=0;
        window_end_time=tw;
        initial_state=3333;  %[1-1.5]^4
        state=initial_state;
        for j=1:window_num
            eeg_start=floor(window_start_time*fs)+1;
            eeg_end=floor(window_end_time*fs)+1;
            if (eeg_end>frame)
                break;
            end
              
            for p=1:choice
                x=eeg(eeg_start:eeg_end,:,i);
      
                Pi(p)=single_cca(x,stimulus(p),fs,eeg_start,eeg_end);
            end
            M=sum(Pi)/choice;
            Prf=Pi./M;
            
            %%%%Q learning%%%%%
            imagine_accept=0;
            if max(Smf)<threshold
                if (Q_table(state,1)==Q_table(state,2))   %same Q value
                    if unifrnd(0,1)<=0.5    %update(use accept)
                        current_reward=update_reward(Prf,REWARD,eeglabel,i);
                        imagine_next_state=next_state_calc(Smf,Prf,partition);
                        imagine_accept=1;
                        Q_table(state,1)=Q_table(state,1)+ALPHA*(current_reward+GAMMA*max(Q_table(imagine_next_state,1),Q_table(imagine_next_state,2))-Q_table(state,1));
                    else 
                        current_reward=update_reward(Prf,REWARD,eeglabel,i);
                        imagine_next_state=state;
                        imagine_accept=0;
                        Q_table(state,2)=Q_table(state,2)+ALPHA*(current_reward+GAMMA*max(Q_table(imagine_next_state,1),Q_table(imagine_next_state,2))-Q_table(state,2));
                    end
                else                       %different Q value
                    compare=[Q_table(state,1),Q_table(state,2)];
                    [~,is_accept]=max(compare);
                    if is_accept==1        %update(use accept)
                        current_reward=update_reward(Prf,REWARD,eeglabel,i);
                        imagine_next_state=next_state_calc(Smf,Prf,partition);
                        imagine_accept=1;
                        Q_table(state,1)=Q_table(state,1)+ALPHA*(current_reward+GAMMA*max(Q_table(imagine_next_state,1),Q_table(imagine_next_state,2))-Q_table(state,1));
                    else
                        current_reward=update_reward(Prf,REWARD,eeglabel,i);
                        imagine_next_state=state;
                        imagine_accept=0;
                        Q_table(state,2)=Q_table(state,2)+ALPHA*(current_reward+GAMMA*max(Q_table(imagine_next_state,1),Q_table(imagine_next_state,2))-Q_table(state,2));
                    end
                end
                
                choose_pro=unifrnd(0,1);
                imagine_next_state=next_state_calc(Smf,Prf,partition);
                if choose_pro<=EPSILION  %real acction
                    if imagine_accept==1
                        Smf=Smf.*Prf;
                        state=imagine_next_state;
                    end
                elseif choose_pro<=0.5
                    Smf=Smf.*Prf;
                    state=imagine_next_state;
                end
            end
            window_start_time=window_start_time+mw;
            window_end_time=window_end_time+mw;
        end
    end
%     if((alpha-0.02)>=0)
%         alpha=alpha-0.02;
%     end
%     if((gamma-0.02)>=0)
%         gamma=gamma-0.02;
%     end
end
new_Q_table=Q_table;
%sum(any(new_Q_table,2))
end

function r=single_cca(x,stimulus,fs,data_start,data_end)
%%signal for single epoch, samples * channels

y1=sin(2*pi*stimulus*(data_start:data_end)/fs);
y2=cos(2*pi*stimulus*(data_start:data_end)/fs);
y3=sin(4*pi*stimulus*(data_start:data_end)/fs);
y4=cos(4*pi*stimulus*(data_start:data_end)/fs);
y5=sin(6*pi*stimulus*(data_start:data_end)/fs);
y6=cos(6*pi*stimulus*(data_start:data_end)/fs);
y=[y1',y2',y3',y4',y5',y6'];%decompose stimulus harmonics
[~,~,r]=canoncorr(x,y);
%r %â€¦â??
r=r(1);
end

function current_reward=update_reward(Pri,REWARD,eeglabel,trail_num)
[~,index]=max(Pri);
if index==eeglabel(trail_num,1)
    current_reward=REWARD;
else
    current_reward=-REWARD;
end
end

function new_state=next_state_calc(smf,prf,partition)
state=zeros(1,4);
smf=smf.*prf;
for i=1:4
    if smf(i)<partition
        state(i)=1;
    elseif smf(i)<(2*partition)
        state(i)=2;
    elseif smf(i)<(3*partition)
        state(i)=3;
    elseif smf(i)<(4*partition)
        state(i)=4;
    elseif smf(i)<(5*partition)
        state(i)=5;        
    elseif smf(i)<(6*partition)
        state(i)=6;        
    else 
        state(i)=7;              
    end
end
new_state=state(1)*1000+state(2)*100+state(3)*10+state(4);
end