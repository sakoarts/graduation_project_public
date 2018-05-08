# graduation_project_public
The public project files belonging to my graduation project for my Master’s in Computer Science and Engineering

## WARNING: None of these files are optimized for public use, readability or anything else for that matter. These are the development files designed for personal use and are far from anything usable in production nor are they suitable for transfer to another engineer. In this state by far not all script are runnable because their missing resources or credentials for external services. 

### /experiment_code
Contains code used in creating all the experiments

#### /experiment_code/etl
Contains all script used to gather the dataset from the external databases, should be usable in this state.

#### /experiment_code/exp
Contains the experimentation scripts, a few different variants usable on different problems. 

#### /experiment_code/tools
Contains extra scripts with tools used by multiple other scripts.

### /notebook_code
Contains the code and notebooks used to analyze the experiments results. These all depend on retrieving these results from a mongoDB. 

#### /notebook_code/process_monog_results
Contains the jupyter notebooks that process and analyze the experiment results from a mongoDB for several of my experiments. 

#### /notebook_code/tools
Contains scripts that are used in the notebooks to do the retrieval, processing and analysis of the experimental results. 
