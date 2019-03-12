function part_hog_feat = compute_part_hog(img, cell_size, feat_type)
switch(feat_type)
    case 1
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = feat(:,:, 1:9); 
    case 2
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = feat(:,:, 10:18);
    case 3
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = feat(:,:, 19:27);
    case 4
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = feat(:,:, 28:31); 
    case 5
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = feat(:,:, 1:18); 
    case 6
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = feat(:,:, 1:27); 
    case 7
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = feat(:,:, 1:9) + feat(:,:,10:18);
    case 8
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = (feat(:,:, 1:9) + feat(:,:,10:18))/2;
    case 9
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = cat(3,feat(:,:, 1:9), feat(:,:,19:27));
    case 10
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = cat(3, feat(:,:, 1:9), feat(:,:,28:31));
    case 11
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = cat(3,feat(:,:, 10:18), feat(:,:,28:31));
    case 12
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = feat(:,:, 19:31);
    case 13
        feat = double(fhog(single(img)/255, cell_size, 9));
        feat(:,:,end) = [];
        part_hog_feat = feat;
    otherwise
        disp('unknown feature type!\n');
end
end