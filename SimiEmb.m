function [Simi] = SimiEmb(pos_fea,neg_fea,pos_number,neg_number,Type)

% Input: 
  
%   - pos_fea: d by N matrix, where d is the feature dimension, N is the
%              total # of frames in positive videos. 

%   - neg_fea: d by M matrix, where d is the feature dimension, M is the
%              total # of frames in negative videos.
%
%  -  pos_num: 1 by P vector, where P is the number of positive videos,
%              pos_num(i) stores the # of frames in the i-th positive video. 
%
%  -  neg_num: 1 by O vector, where O is the number of negative videos, 
%              neg_num(i) stores the # of frames in the i-th negative video. 
%
%  -  Type: 'linear' (by default) or 'L2', metric used to calculate
%            similarity.
% 
%  Output:
%   
%  - Simi: (N+M) by (P+O) matrix, each column denotes the embedding 
%          feature of a video over all frames in the dictionary.
%%
if nargin ==4
    Type = 'linear';
elseif nargin<4
    return;
end

%% Get the average pooled feature of each video
bgn = 0;
edn = 0;
pos_video = zeros(size(pos_fea,1),length(pos_number));
for i = 1:length(pos_number)
    bgn = edn + 1;
    edn = edn + pos_number(i);
    pos_video(:,i) = mean(pos_fea(:,bgn:edn),2);
end

bgn = 0;
edn = 0;
neg_video = zeros(size(neg_fea,1),length(neg_number));
for i = 1:length(neg_number)
    bgn = edn + 1;
    edn = edn + neg_number(i);
    neg_video(:,i) = mean(neg_fea(:,bgn:edn),2);
end

%% Calculate similarity matrix between the frame feature and video feature
Simi = zeros(size(pos_fea,2)+size(neg_fea,2),length(pos_number) + length(neg_number));

if strcmp(Type,'linear') == 1
    Simi = [pos_fea neg_fea]'*[pos_video neg_video];
elseif strcmp(Type,'l2') == 1
    d = L2_distance([pos_fea neg_fea],[pos_video neg_video]);
    sigma = mean(mean(d));
    Simi = exp(-d/sigma);
end

%% Perform normalization
for i = 1:size(Simi,2)
    Simi(:,i) = Simi(:,i)/norm(Simi(:,i),2);
end

Simi = Simi - repmat(mean(Simi,2),[1 size(Simi,2)]);