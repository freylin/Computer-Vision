Please open html/index.html to read the report

1. I integrated non-maxima suppression into function get_interest_points().

2. I set feature_width to 16. Tuning it in proj.m, can get different matching results without doing anything else.

3. The default matching method in this project is applied with kd-tree, you can comment function match_features_kdtree() and uncomment match_features() to use simple matching method.

4. Comment other matching function and uncomment match_features_pca() to apply pca. I have generated two pca basis: pca_32.mat and pca_64.mat and stored them in the code folder. If you want a new basis, set your “dim” and uncomment get_pca_mat() to run it.


Notice:

I have removed all train images (too huge), if you want to get a new pca basis, create a new folder named ”pca_images” in folder “data” and then add enough images into this folder before running get_pca_mat().

Some images shown on the html page are a little small, you can simply zoom the page to see their details or find the source images in html/data/..
