# jamhacks-7-project
Our project made by Steven, Charles, Harry, and Jay

# Setup
Run
```
pip install -r requirements.txt

pip install https://github.com/akTwelve/Mask_RCNN/archive/master.zip

pip uninstall mrcnn
``` 
This installs the necessary packages.
Don't ask me to explain any of this.

Download the [pretrained model](https://drive.google.com/file/d/15ol8TU9pZHemhbpbW3MJxYa-1gheMDN3/view?usp=sharing), move the file (named `mask_rcnn_deepfashion2_0100.h5`) to the `jamhacks-7-project` folder,
and then rename it to `clothing_trained_model.h5`.

Finally, run
```
python clothing_detector.py
```
Congratulations! You survived.
