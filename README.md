# cf-gcn

This is the code for our IJCNLP 17' paper entitled "Estimating Reactions and Recommending Products with Generative Models of Reviews".


Here is the link for the preprocessed dataset and the pretrained models on google drive.
https://drive.google.com/drive/folders/1SCHWdxdhPIQ1PB13dXp-9npNvAJ3uyEO?usp=sharing

Please put the data_dir and save_dir under the same folder of the code. Otherwise, you need to change the directory in the code correspondingly. 

To run the code, you need 
Python==2.7
Keras==0.2 
Theano==1.0

You can train the code by running the script `sh run.sh'.

Similary, you can generate reviews using the provided trained model by running the script `sh generate.sh'.

Note: under folder `lm' contains the pure char-level language model. We first use it to train a language model then load its weight as initialization for cf-gcn.


If you find our code or dataset useful, please cite our paper. Thanks!

```
@inproceedings{Ni2017EstimatingRA,
  title={Estimating Reactions and Recommending Products with Generative Models of Reviews},
  author={Jianmo Ni and Zachary Chase Lipton and Sharad Vikram and Julian McAuley},
  booktitle={IJCNLP},
  year={2017}
}
```
