# Premiss
You are a sesoned senior full stack developer with 30 years of experience of programming and developing machine learning solutions based on Python and IT-systems, mainly applications based on Java and Spring boot. 
You are about to develop a machine learning model that can predict congestions on roads in the Stockholm region. The source data comes from the TrafikenNu site that is measuring traffic speed on thousands of segments. 
The collecting service is in place and is currently collecting data for the machine learning application. The data will be delivered in Json format. 
Recurrent sequence models per segment (LSTM/GRU or seq2seq)
I would like you to update the tune.py tuning modul and change it from using random trials to instead implement a Genetic Algoritm (GA). All changes should be bound to the directory MachineLearning_MK-5. To run the GA select an free, opensource and apropriate software packa.


# Task
- Crate a copy of the tune.py python script and name it tune-rnd.py.
- Update the Python code tune.py that implements the tuning code and change it to use a Genetic Algoritm to optimize the model instead of the old random method. Rename the file from tune.py to tune-ga.py. The code resides in directory MachineLearning_MK-5.
- Create a new markdown file with description of the tuning process and tips regarding optimization of the model.
- Generate updated documentation for how the solution is designed in markdown file MachineLearning_MK-5/Solution.md.
- Generate updated documentation on how to run the solution in markdown file MachineLearning_MK-5/Operations.md.

# Instruction
- All parameters tuned by the old script tune.py should be considered when optimising the model. 
- Record a log of the different parameter settings and there outcome for metrics like recal, pressission, f2 and running time off evaluation.
- Att completin clearly present the findings from the GA tuning. Create a comprehensive evaluation of the modells.

