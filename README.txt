This code was built upon the Keke AI Competition code, which can be found on https://github.com/MasterMilkX/KekeCompetition from Charity, M. (2022).
The code is mainly in JavaScript, but there is one Python file for the creation of graphs for the data.

The files that contain my work include:
1. The file called "File To Run.js" found in the "Keke_JS" directory
2. The file called "orr_LEVELS.json" found in the "Keke_JS\json_levels" directory
3. All files in  the "Keke_JS\Experiments" directory (notably including the "Graph.py" file).

Anything other file is a supporting file from Charity, M. (2022).

Furthermore, any code within the three files which is taken / adapted from other places is commented above the specific functions.
In particular, in the "File To Run.js" file:
1. The OptionsAgent class is adapted from Evans, Smith and Patel (2023).
2. The singleSourceShortestPathBasic, accumulateEndpoints and ImprovedBetweenness functions are adapted from Kling F. (2015) to allow for the addition of weights to the betweenness measure to replicate the agent in Şimşek and Barto (2008).
3. The DegreeMeasure and ClosenessMeasure functions are adapted from Hagberg, Schult and Swart (2008) to allow for a JS version (as it did not exist on Kling F. (2015)).
4. getChildNode1 and getChildren functions are adapted from Charity, M. (2022), but do not rely on parent node.

In the Python file "Graph.py" in the , the only code taken from another source was the function moving_average, taken from Saxenaanjali239 (2021).


You might also need to install a few libraries to run the code fully being:
1. JSNetworkX from Kling, F. (2015) - This is used to create the state transition graph of the level in order to create the options.
2. lodash from Anon. (2016) - This is used to deepclone the state to avoid storing a state's parent.

To install these go to the terminal and type: 
npm install jsnetworkx
npm i -g npm
npm i --save lodash

If you wish to create new levels, you could do it in the "orr_LEVELS.json" file, or create your own json file.

To run the agents, please enter the "File To Run.js" file.
There, change the variables at the top of the file to suit your requirements:
1. Change the agent_type string variable to the agent you wish to run, namely "degree", "closeness", "betweenness", "improvedBetweenness", "handcrafted" or "q_learning" (currently set to "improvedBetweenness"). Note that the handcrafted agent was designed for level 1 and 2, so it might crash the program if run on a different level.
2. Change the lvl_set_name string variable to your json's file name (currently set to "orr_levels").
3. Change the lvl_num integer variable to the level id in your json file (currently set to 1, which is the level the experiments were run on).
4. Change the file_saves boolean variable to true if you wish the data to be saved, and false otherwise (currently set to true).
5. Change the num_agents integer variable to the number of trials you wish to repeat for (currently set to 100).
6. Change the num_episodes integer variable to the number of episodes you wish to run for (currently set to 15000).

Then run the code by running the file. On windows you can navigate to the directory on command prompt and then use the command:
nodemon "File To Run.js" 

or instead if you don't have nodemon then use:
node "File To Run.js".

Get the files (in particular the training results file) into an experiment directory in a similar manner to the experiments done.

To create the graph from results, please enter the "Graph.py" file.
There, change the variables at the top of the file to suit your requirements:
1. Chage the cur_dir string variable to the file's directory's full path (this NEEDS to change to work on other computers) and inside_dir string variable to the folder containing all test results.
2. Change the experiment integer variable to either 1, 2 or 3 depending on the experiment results you wish to replicate (3 being the scaled-down experiment in appendix) - (currently set to 2)
3. Change the moving_average boolean variable to True or False depending on whether you want a smoothed or raw graph respectively (currently set to True).

Then run the code by running the file. On windows you can navigate to the directory on command prompt and then use the command:
python Graph.py 

Bibliography:

Anon. (2016). lodash. [online] GitHub. Available at: https://github.com/lodash/lodash

Charity, M. (2022). Keke AI Competition. [online] GitHub. Available at: https://github.com/MasterMilkX/KekeCompetition

Evans, J., Smith T., Patel A. (2023). SimpleOptions. [online] GitHub. Available at: https://github.com/Ueva/BaRL-SimpleOptions

Hagberg, A. A., Schult, D. A., Swart P. J. , “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008

Kling, F. (2015). JSNetworkX. [online] GitHub. Available at: https://github.com/fkling/JSNetworkX

Saxenaanjali239 (2021). How to Calculate Moving Averages in Python? [online] GeeksforGeeks. Available at: https://www.geeksforgeeks.org/how-to-calculate-moving-averages-in-python/

Şimşek, Ö. and Barto, A.G., 2008. Skill characterization based on betweenness [Online]. In: D. Koller, D. Schuurmans, Y. Bengio and L. Bottou,
eds. Advances in neural information processing systems. Curran Associates, Inc. Available from: https://proceedings.neurips.cc/paper_files/paper/2008/file/934815ad542a4a7c5e8a2dfa04fea9f5-Paper.pdf

