function [time,temp_confusion_matrix,accuracy,accept_num]=Q_test(tw,mw,threshold,partition,train_time,fs,stimulus,eeg,eeglabel,Q_table)
%[accuracy,accept_num]=Q_test(3,0.2,1.2,10,256,[6 7 8 9],EEGdata,EEGdatalabel)
%[accuracy,accept_num]=Q_test(3,0.2,1.2,10,256,[6 7 8 9],EEGdata,EEGdatalabel,fit_Q_table)
%[accuracy,accept_num]=Q_test(3,0.2,1.2,10,256,[6 7 8 9],EEGdata,EEGdatalabel,Q_sd_train(3,0.2,1.2,10,256,[6 7 8 9],EEGdata,EEGdatalabel,zeros(7777,2)))
%Q_table = xlsread('output2.xlsx');
[channel,frame,trail]=size(eeg);
eeg=permute(eeg,[2,1,3]);   %frame*channel*trail
[~,choice]=size(stimulus);
window_num=floor((train_time-tw)/mw+1);
correct_num=0;
accept_num=0;
temp_confusion_matrix=zeros(4,4);
EPSILION=1;
time=0;
partition=(threshold-0.3)/6;
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
                    if unifrnd(0,1)<=1  %update(use accept)
                        imagine_accept=1;
                    end
                else                       %different Q value
                    compare=[Q_table(state,1),Q_table(state,2)];
                    [~,is_accept]=max(compare);
                    if is_accept==1        %update(use accept)
                        imagine_accept=1;
                    end
                end
                choose_pro=unifrnd(0,1);
                imagine_next_state=next_state_calc(Smf,Prf,partition);
                if unifrnd(0,1)<=EPSILION  %real acction
                    if imagine_accept==1
                        Smf=Smf.*Prf;
                    end
                    if (Q_table(state,1)<Q_table(state,2))&&(choose_pro<=0)
                        Smf=Smf.*Prf;
                    end
                state=imagine_next_state;
                end
            end
            [Smf_max,sd_index]=max(Smf);
            if Smf_max>=threshold
                %threshold
                %Smf
                temp_confusion_matrix(sd_index,eeglabel(i))=temp_confusion_matrix(sd_index,eeglabel(i))+1;
                accept_num=accept_num+1;
                time=time+eeg_end/fs;
                if sd_index==eeglabel(i)
                    correct_num=correct_num+1;
                end
                break;
            end
            %Smf
            window_start_time=window_start_time+mw;
            window_end_time=window_end_time+mw;
        end
%         if Smf_max<threshold
%             temp_confusion_matrix(sd_index,eeglabel(i))=temp_confusion_matrix(sd_index,eeglabel(i))+1;
%                 accept_num=accept_num+1;
%                 time=time+eeg_end/fs;
%                 if sd_index==eeglabel(i)
%                     correct_num=correct_num+1;
%                 end
%         end
        
%         [Smf_max,sd_index]=max(Smf);
%         if Smf_max>=threshold
%             temp_confusion_matrix(sd_index,eeglabel(i))=temp_confusion_matrix(sd_index,eeglabel(i))+1;
%             accept_num=accept_num+1;
%             time=time+eeg_end/fs;
%             if sd_index==eeglabel(i)
%                 correct_num=correct_num+1;
%             end
%         end
end
if accept_num==0
    accuracy=-1;
else
    accuracy=correct_num/accept_num;
end
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

function new_state=next_state_calc(smf,prf,partition)
state=zeros(1,4);
smf=smf.*prf;
for i=1:4
    if smf(i)<(0.3+partition)
        state(i)=1;
    elseif smf(i)<(0.3+2*partition)
        state(i)=2;
    elseif smf(i)<(0.3+3*partition)
        state(i)=3;
    elseif smf(i)<(0.3+4*partition)
        state(i)=4;
    elseif smf(i)<(0.3+5*partition)
        state(i)=5;        
    elseif smf(i)<(0.3+6*partition)
        state(i)=6;        
    else 
        state(i)=7;              
    end
end
new_state=state(1)*1000+state(2)*100+state(3)*10+state(4);
end