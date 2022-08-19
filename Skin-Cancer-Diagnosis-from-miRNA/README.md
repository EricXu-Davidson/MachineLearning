# Skin Cancer Diagnosis from miRNA Data
Author: Eric Xu & Kostas Mateer

# data_exploration.py:
Found in the `/data` folder.

This file was used to explore the data and compile it into a single csv file.

Finds and removes features that are 0 valued.

No need to run this file as it will create a bunch of images that are already
    stored in `/presentation` or are in the write up itself.

# plots.py:
Found in the `/src` folder.

This file was used to make various plots and mess around with the graphviz package.

No need to run this file as it will create plots and images that are already
    stored in `/presentation` or `/write up` folder.

# decision_tree_model.py:
Found in the `/src` folder.

*Need Graphviz package installed*
------ Homebrew works great to install it ------ 

This file contains our decision tree model.

When running this file it runs through a 99 models of decision trees with varying depth.
It will also create a png file for each model, so may want to move that out of the for loop
before running.

It was found that a tree depth of 7 was found to be most accurate (check figure in write up):
    `max_depth = 1` `accuracy score = 0.568`
    `max_depth = 2` `accuracy score = 0.744`
    `max_depth = 7` `accuracy score = 0.938`
    `max_depth = 99` `accuracy score = 0.917`

# svm_model.py:
Found in the `/src` folder.

This file contains our svm model.

While running this file it creates different classifiers for svm model building, K-fold validation, gridsearch.

The result of gridsearch shows that linear model is the most accurate model compared with polynomial model of degree 2 and 3. And different regulazation values do not influence the accuracy of linear model. Check more details in write up.

# write up
The write up can be found in the `/write up` folder.

Contains png files of tree models and the tree depth vs accuracy score.

If you want full image of tree go to `/write up` folder and select `tree_model.png` because had to put only a small
portion of the full tree to fit into write up.

# CSC371_project3.pdf:
Found in the `/write up` folder.

This pdf is the write up document for this project.


