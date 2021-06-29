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
     
     
    


