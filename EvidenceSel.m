function [w] = EvidenceSel(pos_fea, neg_fea, num_frame_per_pos_vid, num_frame_per_neg_vid)

% Input: 

%  - pos_fea : (d by N matrix) where d is feature dimension, N is the number of
%              frames in all positive videos. For each video, the frames are stacked
%              in this matrix according to their respective temporal order in the video. 
                  
%  - neg_fea : (d by M matrix) where d is the feature dimension, M is the number 
%               of frames in all negative videos. For each video, the frames are stacked 
%               in this matrix according to their respective temporal order in the video.         
     
% -  num_frame_per_pos_vid :  1 by P vector, num_frame_per_pos_vid(i) indicates the #
%                             of frames in the i-th positive video, P is the total # of
%                             positive videos. 
% -  num_frame_per_neg_vid :  1 by L vector, num_frame_per_neg_vid(i) indicates the # of
%                             frames in the i-th negative video, L is the total # 
%                             of negative videos. 

% Output:
%   - w : 1 by (N+M) evidence selective parameter vector in which each entry indicates 
%         the importance of the corresponding frame in a video. 

%% Assign weak label of each training video to its individual frames
y = [ones(length(num_frame_per_pos_vid),1); -1*ones(length(num_frame_per_neg_vid),1)]; 

%% Perform Similarity embedding 
fprintf('Perform video similarity embedding..\n');
[Simi] = SimiEmb(pos_fea, neg_fea, num_frame_per_pos_vid, num_frame_per_neg_vid, 'l2');

%% Construct temporal consistency matrix 
fprintf('Fill in the temporal consistency relations of frames...\n');
count = 0;
S = zeros(size(Simi,1),size(Simi,1)); % Laplacian Graph
for i = 1: length(num_frame_per_pos_vid)
    for j = 1: num_frame_per_pos_vid(i) - 1
        count = count + 1;
        S(count,count+1) = 1;
        S(count+1,count) = 1;
    end
    count = count + 1;
end
count = count - 1;

for i = 1:length(num_frame_per_neg_vid)
    for j = 1:num_frame_per_neg_vid(i) - 1
        count = count + 1;
        S(count,count+1) = 1;
        S(count+1,count) = 1;
    end
    count = count + 1;
end

D = diag(sum(S,2));
S = D-S;

%% Solve ADMM 
fprintf('Select frames with ADMM procedure...\n');
X = [repmat(y,[1,size(Simi,1)]).*Simi'];
lambda = 0.0005;
options.mu = 10^(-3);  
options.nbitermax = 30;
options.stopobjratio = 10^(-3);
C = 0.001;
[w,obj] = Solve_ADMM(X,S,lambda,C,options);

end