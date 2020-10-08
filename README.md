# CapIRMA

1. Download the project using git clone from the terminal or as a zip using the gui. 

2. All dependencies needs to be installed along with pytorch

3. Download the data using the urls provided in data/data.txt.  Save the unzipped file in the data folder.

4. create a folder newaug inside the project folder

5. Run the code imfileaug.py to generate augmented data. You can vary number of images (default is 1000) , alpha (default 1.01-1.1), beta (default 0-24) and rotation angles(default -10 to 10 degrees) as per your requirement.

6. Create the training and test files 

7. Run the training script using
sh train.sh
This will train all 12 networks described in the paper one by one. If you want to train a single network, comment out all other lines. For example if you just want to train the best performing one use
python3 main.py --pc 32 --nets 3 --primary-unit-size 3872 > f32_3.txt
mv     results/trained_model/model_epoch_15.pth  results/trained_model/model_epoch_15_32_3.pth
The training script will not only train the network, but also compute the classification scores and save it in the file f32_3.

8. For testing using a trained model you can run. You can provide the parameters (pc,net,primary-unit size) in the same way as the training script. Alternatively you can change them in the argument list inside the code. Also provide the appropriate pretrained model in line 343.
python3 test.py

