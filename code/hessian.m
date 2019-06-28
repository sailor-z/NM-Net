function [frames, patches] = hessian(img)
max_num = 2000;
addpath ('../vlfeat-0.9.21/toolbox/');
vl_setup;
im=imread(img);
rgb=size(im);
if numel(rgb)>2
    imgs = im2single(rgb2gray(im));
else
    imgs=im2single(im); 
end

[frame, patche, info] = vl_covdet(imgs,'method', 'HessianLaplace','descriptor', 'sift','estimateAffineShape', true,'estimateOrientation', true);
un_frams = frame(1:2,:)';

[un, in] = unique(un_frams, 'rows');
frame = frame(:,in);
patche = patche(:,in);

frames = zeros(size(frame,1), max_num);
patches = zeros(size(patche,1), max_num);

if size(frame,2) < max_num
    for i=1:size(frame,2)
        frames(:, i) = frame(:, i);
        patches(:, i) = patche(:, i);
    end
    fprintf('Cant detect enough keypoint!!\n');
else
    info.peakScores = info.peakScores(:,in);
    info.edgeScores = info.edgeScores(:,in);
    info.orientationScore = info.orientationScore(:,in);
    info.laplacianScaleScore = info.laplacianScaleScore(:,in);
    sum=info.peakScores+info.edgeScores+info.orientationScore+info.laplacianScaleScore;
    sort_sum= sort(sum,'descend');
    if size(frame,2) > max_num
        sort_sum=sort_sum(1:max_num);
    end

    for i=1:size(sort_sum,2)
        index=find(sum==sort_sum(i),1);
        frames(:,i)=frame(:,index);
        patches(:,i)=patche(:,index);
    end
end


