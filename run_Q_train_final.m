function run_Q_train_final()
%run_Q_train(30,256,[6 7 8 9])
train_time=10;fs=256;stimulus=[6 7 8 9];
root_address='E:\code\SSVEP_BCI\';
sub_address=[root_address,'EEGdata_cipher\EEGdata_cipher\'];
max_accuracy=-1;
fit_tw=0;
fit_mw=0;
fit_threshold=0;
fit_partition=0;
fit_alpha=0;
fit_gamma=0;
trail_num=12;
flag=0;
v_index=1;
%用于存储每次交叉验证后的max_accuracy，共五次交叉验证，所以有五列，每行存储不同的元素，共七行
%'max_accuracy','fit_tw','fit_mw','fit_threshold','fit_partition','fit_alpha','fit_gamma'

for person_num=1:3
    t=0;
    tic
    alpha=0.5;gamma=0.5;fit_alpha=0.5;fit_gamma=0.5;
    confusion_matrix=zeros(4,4);accept_count=0;fit_Q_table=zeros(7777,2);temp_array=zeros(7,5);
    fit_confusion_matrix=zeros(4,4);new_matrix=zeros(5,2);count=1;
    Q_matrix=zeros(7777,2,5);
    max_accuracy=-1;fit_accept=0;
    flag=0;
    person_num
    %address=[sub_address,num2str(person_num),'\',num2str(train_table_num),'.mat'];
    %load(address)
    tw=fit_tw;mw=fit_mw;threshold=fit_threshold;partition=fit_partition;
    if (person_num==7)
        mw=0.8;
    elseif(person_num==9)
        mw=0.6;
    elseif(person_num==12)
        tw=5;threshold=2.4;
    elseif(person_num==13)
        threshold=1.8;
    elseif(person_num==15)
        tw=1;
    end
    for episode=10:10:100
        %for v_index=1:5
        temp_accuracy=0;temp_accept_num=0;Q_table=zeros(7777,2);temp_confusion_matrix=zeros(4,4);
        max_accuracy=-1;
        for train_table_num=1:5
            %if (train_table_num ==v_index)continue;end%train table
            address=[sub_address,num2str(person_num),'\',num2str(train_table_num),'.mat'];
            load(address);
            eeg=EEGdata;
            eeglabel=EEGdatalabel;
            Q_table=Q_sd_train_f(episode,tw,mw,threshold,partition,alpha,gamma,train_time,fs,stimulus,eeg,eeglabel,Q_table);%训练
        end
        %                     output4=num2cell(Q_table);
        %                     output4=[{'First','Second'};output4];
        %                     result=xlswrite('output4.xlsx',output4);
        for v_index=1:5%交叉验证
            address=[sub_address,num2str(person_num),'\',num2str(v_index),'.mat'];
            load(address);
            eeg=EEGdata;
            eeglabel=EEGdatalabel;
            [confusion_matrix,accuracy,accept_num]=Q_test(tw,mw,threshold,partition,train_time,fs,stimulus,eeg,eeglabel,Q_table);%测试存储结果
            temp_confusion_matrix=temp_confusion_matrix+confusion_matrix;
            temp_accuracy=temp_accuracy+accuracy*accept_num;
            temp_accept_num=temp_accept_num+accept_num;
        end
        %end
        accuracy=temp_accuracy/60;accept_num=temp_accept_num;
        if(accuracy>max_accuracy)&&(accept_num>=48)%从结果中挑选最佳结果保存
            max_accuracy=accuracy
            accept_num
            fit_tw=tw
            fit_mw=mw
            fit_threshold=threshold
            fit_partition=partition
            fit_Q_table=Q_table;
            fit_confusion_matrix=temp_confusion_matrix;
        end
        new_matrix(count,1)=episode;
        new_matrix(count,2)=max_accuracy;
        Q_matrix(:,:,count)=fit_Q_table;
        count=count+1;
    end
    t=toc;
    save_name=[root_address,'Q_table\final_1000\',num2str(person_num),'_episode.mat'];
    save(save_name,'new_matrix','Q_matrix');%保存
end
end
%     output2=num2cell(fit_Q_table);
%     output2=[{'First','Second'};output2];
%     result=xlswrite('output8.xlsx',output2);

%train_time=10;fs=256;stimulus=[6 7 8 9];root_address='F:\codes\SSVEP_BCI\';sub_address=[root_address,'EEGdata_cipher\EEGdata_cipher\'];max_accuracy=-1;fit_Q_table=zeros(7777,2);person_num=8;
%Q_table = xlsread('output8.xlsx','Sheet1');fit_tw=5;fit_mw=1;fit_threshold=1.8;fit_partition=0.4;fit_alpha=0;fit_gamma=0;trail_num=12;
%max_accuracy=-1;fit2_Q_table=zeros(7777,2);
%     for alpha=0:0.2:1       %RL train
%         for gamma=0:0.2:1
%             Q_table=fit_Q_table;temp_confusion_matrix=zeros(4,4);temp_accuracy=0;temp_accept_num=0;
%             for v_index=1:5
%                 for train_table_num=1:5
%                     if (train_table_num ==v_index)continue;end
%                     address=[sub_address,num2str(person_num),'/',num2str(train_table_num),'.mat'];
%                     load(address);
%                     eeg=EEGdata;
%                     eeglabel=EEGdatalabel;
%                     Q_table=Q_rl_train(fit_tw,fit_mw,fit_threshold,fit_partition,alpha,gamma,train_time,fs,stimulus,eeg,eeglabel,Q_table);
%                 end
%                 address=[sub_address,num2str(person_num),'/',num2str(v_index),'.mat'];
%                 load(address);
%                 eeg=EEGdata;
%                 eeglabel=EEGdatalabel;
%                 [confusion_matrix,accuracy,accept_num]=Q_test(fit_tw,fit_mw,fit_threshold,fit_partition,train_time,fs,stimulus,eeg,eeglabel,Q_table);
%                 temp_confusion_matrix=temp_confusion_matrix+confusion_matrix;
%                 temp_accuracy=temp_accuracy+accuracy;
%                 temp_accept_num=temp_accept_num+accept_num;
%             end
%             accuracy=temp_accuracy/5;accept_num=temp_accept_num/5;
%             if (accuracy>max_accuracy)&&(accept_num>=0.8*trail_num)
%                 max_accuracy=accuracy
%                 fit_alpha=alpha;
%                 fit_gamma=gamma;
%                 fit_accept=accept_num;
%                 fit_confusion_matrix=temp_confusion_matrix;
%                 fit2_Q_table=Q_table;
%             end
%         end
%     end