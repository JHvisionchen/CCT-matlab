function display = display_feature_type(feat_type)
switch(feat_type)
    case 1
        disp('Only the first 1:9 features of HOG!');
    case 2
        disp('Only the first 10:18 features of HOG!');
    case 3
        disp('Only the  19:27 features of HOG!');
    case 4
        disp('Only the  28:31 features of HOG!');
    case 5
        disp('Only the 1:18 features of HOG!'); 
    case 6
        disp('Only the 1:27 features of HOG!');
    case 7
        disp(' feature 1:9 plus feature 10:18!');
    case 8
        disp('feature 1:9 plus feature 10:18, and divide 2!');
    case 9
        disp('feature 1:9 and feature 19:27!');
    case 10
        disp('feature 1:9 and feature 28:31!');
    case 11
        disp('feature 10:18 and feature 28:31!');
    case 12
        disp('feature 19:31!');
    case 13
        disp('all features of HOG!');
    otherwise
        disp('unknown feature type!');
end
end