Training
=============

To train the model run the following command. You can resume training by providing a checkpoint.
```
$ python train.py <checkpoint_output_directory> [<checkpoint_to_load.ckpt>]
```

To test streaming inference execute the following command.
You will see exactly the same output from processing in chunks as from processing everything at once. 
```
$ python streaming_inference.py <inputfile.ckpt>
```

To convert the model into a CoreML model execute the following command.
```
$ python coreml.py <inputfile.ckpt> <outputfile.mlmodel>
```