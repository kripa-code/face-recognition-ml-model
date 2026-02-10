# ML project to learn knn, using OpenCV, Scikit-Learn, Numpy

how to run:
    1. inside terminal/command promt run the following command:
            sh run.sh
    2. enter ur name in the terminal when asked
    3. camera window will open once and capture ur face and close on its own
    4. model will be trained 
    5. camera window will open again and display ur name on top of ur face
    6. press "q" to exit


libraries used:
    1. openvc - camera operation, face detection
    2. sckikit-learn - to create ml model
    3. numpy  - for image processing and cleaning


working steps of the project:
    once u run the run.sh file
    1.
        a. a virtual env is created
        b. dependencies - pip, opencv, scikit, numpy are installed
    2. 
        a. collect faces is run
        b. opens camera
        c. captures that frame
    3. 
        a. makes the pic b&w
        b. crops the face
        c. stores it in the dataset
    4.
        a. facial data is then taken and flattened
        b. stored with label of the person name
        c. flattened data and labels are then fed to the knn model
        d. the model is stored as a binary file
        e. k value is hard-coded as 3
    5.
        a. saved ml model is opened
        b. camera is opened
        c. frames are captured live
        d. faces are detected and names are predicted using the ml model.
    
