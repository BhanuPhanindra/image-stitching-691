I1 = imread('\Users\goran\Desktop\Computer Vision\HW2\matlab\hw2_data\uttower_left.JPG');
I1 = rgb2gray(I1);

I2 = imread('\Users\goran\Desktop\Computer Vision\HW2\matlab\hw2_data\uttower_right.JPG');
I2 = rgb2gray(I2);

[cim1, corner_r1, corner_c1] = harris(I1, 1.2, 5000, 3, 0);
corner1 = [corner_c1, corner_r1];
%imshow(cim1);
[cim2, corner_r2, corner_c2] = harris(I2, 1.2, 5000, 3, 0);
corner2 = [corner_c2, corner_r2];

sift1 = find_sift(I1, corner1, 1.5);
sift2 = find_sift(I2, corner2, 1.5);

euclidean_matrix = dist2(sift1, sift2);
%%
[~, sortIndex] = sort(euclidean_matrix(:), 'ascend');  % Sort the values in ascending order
tmp1 = size(euclidean_matrix);
matches = get_matches(sortIndex, tmp1(2), 10, corner1, corner2);

function matches = get_matches(sortIndex, col, count, corner1, corner2)
    matches = zeros(count, 2, 2);
    for i = 0:count
        i1_des = floor(sortIndex(i) -1 / col) + 1;
        i2_des = mod(sortIndex(i) - 1, col) + 1;
        matches(i) = [ [corner1(i1_des, 2), corner1(i1_des, 2)], [corner2(i2_des, 2) , corner2(i2_des, 1)] ];
    end
end


%https://www.mathworks.com/help/matlab/ref/mink.html
%bestmatch1 = mink(euclidean_matrix, 1, 2);

%{
get_descriptors(I1, corner_r1, corner_c1, 3);
function f = get_descriptors(I, r, c, neighborhood)
    f = prod(1:n);
    for n = 1:size(r)(1)
        start = [r(n) - neighborhood, c(n) - neighborhood;
        for i = start(1):1:start(1) + (2 * neighborhood) + 1
            
   end
%}