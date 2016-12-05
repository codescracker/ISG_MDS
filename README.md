To run this Project:

1. Use python 3.5, install packages and libraries: pandas, numpy, scipy, sklearn, flask.
2. In the terminal, navigate to the dictionary of project, and type `python app.py` to start the lcoal server. 
3. In the browser, type `localhost:5000`. 
4. There are 3 plots, Bar Chart is used to show the weight of each feature in current round, Parallel Cooridinates plot 
used to show the value of each feature of each observation. Scatter Plot is used to display the result the dimensionality 
reduction of original dataset, if you are not satisifited with their position, you can drag the dots in the Scatter Plot
to the position that you think they should be. Then click the button `Post New Position `, after this, click the button 
`Calculate New Weight`, this step maybe gonna take a long time for computation, once the computation is finished,
 the brwoser will notify you. To get the new result of dimensionality reduction that is optimized by your training, 
 click button `Draw New Plots`, then the browser will show you the updated Scatter Plots and Bar Chart which is based 
 on updated dimensionality reduction and weights of features. 
5. To get more details of status, please check the terminal, it will print out every important step and values. 
6. For the convinience of use, later maybe I will push the python virtual environment to repository, so that users no
 longer need to worry about python environment issues. 

