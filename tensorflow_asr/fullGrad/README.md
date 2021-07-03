So basically, till now we have seen, that a model is made
then its iterated over the modules, 
for each layer, a back hook is attaached 
and biases are extracted out of each layer(conv2d, linear, batchnorm)

So in fullGradExtractor 3 things are extracted->
  1) biases of each layer, 
  2) grad_handles(hooks)
  3) featre grad(empty)
  
Now we will move to checkCompleteness

 1) first we got output from model's forward pass
 2) Then we obtain input_grad and bias_grad from Decomp fx
 3) Finally we assert that [(input*input_grad) + sigma(bias_gradients)] == model_output
     
     
Now we will see what is Decomp fx:

 1) It performs backprop with resp to the output of the model
 d(output)/d(input) and stores these grads in grad_feature.
 2) Then these grad_features are multiplied by bias(iterating)
 3) Finally we return input_grad and these (bias*feature_grad)
 4) cool thing is that hook, when we have this statement -
    1) `input_gradients = torch.autograd.grad(outputs = output_scalar, inputs = x)[0]`
    2) the feature_grad automatically stores the output gradients(during backprop, from that hook callable object)
    3) remember to clear the model.zero_grad() if using pytorch
        
===============================================================================

#Part 2 ( running the actual saliency function)

1) A little noise is added to the input image
2) Decomp fx is called on this image which gives input_grad and bias_grad
3) Finally some gradient accumulation and post processing. 
 

==============================================================================

#Part 3( writing the code for tensorflow)

    'Learnings -'
  
1) created a small tensorflow subclassing model, identical to the small pytorch
model.
2) Observations 
    1) the model had 5 layers in a sequence, but when i compiled it the trainable variables 
    was as list of size 10
    So each layer is followed by its bias layer. 
    2) model.layers resulted in 5 layers.
    3) while iterating over model.layers:
        layer.bias will give you the bias and its shape would be same as that 
        of the output of that layer
    4) Beware that only dense and conv layers have bias, maxpool, and flatten dont.
    5) in the pytorch model - output_grad is from each layer. So we cant just simply take outputs from
     conv_blocks. We need to dig deeper and accumulate them from individual layers(obviously conv layers)
     Then taking biases from those layers, we need to perform that bias*grad operation.

# Part 4(integrated gradients IG)
Source = 
    https://towardsdatascience.com/understanding-deep-learning-models-with-integrated-gradients-24ddce643dbf

1) Sir adviced to read this.
2) Fundamentally it seems to be similar, 
`input feature importance that contributes to the model's prediction`

3) computes the gradient of the model’s prediction output to its input features and requires no modification to the original deep neural network.
4) Its built on 2 axioms(again very similar to what saliency s based on)
5) saliency defined completeness, and in IG we have Sensitivity
6) Sensitivity = we establish a Baseline image as a starting point. We then build a sequence of images which we interpolate from a baseline image to the actual image to calculate the integrated gradients.
7) 2nd is Implementation invariance
    1) Lets say we have 2 functionally equivalent networks(when their outputs are equal for all inputs despite having very different implementations. )
    2)  IV is satisfied when two functionally equivalent networks have identical attributions for the same input image and the baseline image.
    
8) Lets calculate and visualise IG
    
    1) Step 1: Start from the baseline where baseline can be a black 
    image whose pixel values are all zero or an all-white image, or a random image. 
    Baseline input is one where the prediction is neutral and is central to any explanation method and 
    visualizing pixel feature importances.
    
    2) Step 2: Generate a linear interpolation between the baseline and the original image. 
    Interpolated images are small steps(α) in the feature space between your baseline and 
    input image and consistently increases with each interpolated image’s intensity.
       
       ` 
         x = x' + á(x-x') 
         1) á -> interpolation constant to perturbed features
         2) x -> input image tensor
         3) x'-> baseline image tensor`
      
    3) Step 3: Calculate gradients to measure the relationship between changes to a feature and 
    changes in the model’s predictions.
    
        1) So basically gradient informs which pixel has max effect on model's predicted class probabilites
        2) For interpolated images, gradient is calculated of the predicted logit(probability) wrt to input(image)
    
    4) Step 4: Compute the numerical approximation through averaging gradients
    
    5) Step 5: Scale the IG to input image and overlay to see the effects of attribution masks
    
 
 
#My leanrings

    1) Norms = basically size of a vector(l1(manhatten), l2(euclidean))
     
    


