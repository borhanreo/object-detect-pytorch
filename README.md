## Pytorch train a model for object detection in an image
#### Step 1: Description
 
 - At first you need to decided how may object you need to detect. Here i used 2 object car and bottle
 - Object name write into a json 'cat_to_name.json' file    {"1":"bottle","2":"Car"} 
 - We need to arrange/collect train image. all image will be in data folder like this. data/train/1/images_00000.jpg here 1 for all bottle image and another data/train/2/images_00000.jpg car images  

#### Step 2: Initialize
 
    #cat_to_name_file
    {"1":"bottle","2":"Car"}
    #Change some parameter it's depend on imput class (we used 2) and output (we used 2) 
    num_features = model.classifier.in_features
    classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(num_features, 512)), #### change input
                                  ('relu', nn.ReLU()),
                                  ('drpot', nn.Dropout(p=0.5)),
                                  ('hidden', nn.Linear(512, 100)),                       
                                  ('fc2', nn.Linear(100, 2)), ##### Change output
                                  ('output', nn.LogSoftmax(dim=1)),
                                  ]))
    
    model.classifier = classifier    

##### Save the checkpoint

    # TODO: Save the checkpoint 
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    model.epochs = num_epochs
    checkpoint = {'input_size': [3, 224, 224],
                     'batch_size': dataloaders['train'].batch_size,
                      'output_size': 2,
                      'state_dict': model.state_dict(),
                      'optimizer_dict':optimizer.state_dict(),
                      'class_to_idx': model.class_to_idx,
                      'epoch': model.epochs}
    torch.save(checkpoint, 'object-detect.pth')    
    
So we get a pth file object-detect.pth file now we are ready to test our model

    python predict.py    
    
    
    
#### None issue

There are no object-detect.pth file. so we need to train and save the file.    