# Adding New Model

Steps to be followed
--------------------
1. Go to the [models](/models/) subdirectory.
2. Create a new file for you model, create a new class which extends [Bare Class](/models/bare.py) with your model logic.
3. Go [here](/models/__init__.py), add a line to import your model class
4. Run the very basic [test](/test_model.py) in debug mode to make sure that everything works as you intended it to be. Refer to the lines 9 and 32 to get the list of changes to be made for your model to be added.
5. In the [train](/train.py) file:
    1. Add your model keyword in line 29.
    1. Add your model specific hyper-parameter defaults from line 35. 
    2. After line 79, add a simple if statement to create your model with the keyword.
6. Voila! You are done.