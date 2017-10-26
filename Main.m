addpath(genpath('./SLEP_package_4.1/'));

%% Synthesize some video features as toy examples to run the program:
%  Assume each video contains 10 frames, generate 30 positive videos and 20
%  negative videos. After this, there are 300 frames in positive videos and 
%  200 frames in the negative videos. Each frame is represented as a 3000 
%  dimensional feature vector.  
 
 num_pos_vid = 30; 
 num_neg_vid = 20; 
 fea_dim = 3000; 

 pos_fea = rand(fea_dim, num_pos_vid*10);
 neg_fea  = rand(fea_dim, num_neg_vid*10);

%% Store the number of frames in each positive/negative video into two vectors. 
num_frame_per_pos_vid = 10*ones(1,num_pos_vid);
num_frame_per_neg_vid = 10*ones(1,num_neg_vid);

%% perform frame selection. 
[w] = EvidenceSel(pos_fea, neg_fea, num_frame_per_pos_vid, num_frame_per_neg_vid); 

%% Rank the optimized w and choose the frames with the largest w values within each video as 
%  the selected key evidences. Please customize according to the needs. 
[selection, idx] = sort(w);