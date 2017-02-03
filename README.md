Overall Project Design:
My inclination was to use transfer learning, via either the VGG or one of Google's models (different reasons depending on which of the two Google nets I could've chosen), but ultimately I wanted the challenge of replicating NVIDIA's frankly amazing results --if only in a simulator. With that in mind, I chose a very nearly identical architecture as detailed below, with personal choices --usually made empirically- where the NVIDIA paper had been sparse on detail (regularization, for one). 

(My initial Keras code was modified from one of their convnet samples). 

Data Collection:
I collected my data from the 50Hz simulator and a PS4 controller, using only the first track. I recorded eight laps forward, five in reverse, and a series of broken recoveries (one in each direction). In order to collect recovery data, I would turn off the recording, drive to an undesireable angle, adjust the steering to the appropriate angle to correct the car's path, and then reengage the data recording as I drove that back through to the center of the road. All told, I collected about 65000 honest images. I had originally used a similar but smaller set from the default generator cat'ed to the Udacity set, but I found the results staggeringly better with this latter data set, and kind of lament not having put more thought into my data collection from the outset (despit having Paul Heraty's list of tips from the very beginning).



Due to the naturally imbalanced data (overwhelming fraction of the steering angles were within +/- 0.1 degrees, see above) I had to adjust my data artificially. First, in line with the NVIDIA paper, I used the side cameras to simulate quasi-recovery angles, adjusting the labels appropriately (this was done at random, in the data generator). I was also strongly inspired by Vivek's work with augmentation of existing images via brightness adjustment, affine transformations, and synthetic shadows to create a more varied, larger training set from the extent data, I rewrote my own functions, having some slightly different tastes in python, but borrowed his double m-grid method for shadow augmentation (really an awesome use of numpy, I definitely tucked that one away for later use ;-) ). I also selectively allowed images within the +/- 0.1, typically only 5 percent of the times that such an image was drawn. I wrote a data generator to pair with the Keras fit-generator function to avoid having to try to downsize my training data or go crazy trying to optimize fitting the whole thing in at once.

Architecture:
Again, I chose to try to (largely, anyway) replcicate the NVIDIA end-to-end paper results. To that end, I used an architecture consisting of:
* 1 normalization "lambda" layer; mean-normalization and pseudo-centering
* 1 color-remapping layer; 3 1x1 kernels; 1x1 stride; no activation; no regularization
* 1 convolutional layer; 24 5x5 kernels; 2x2 stride; ReLU activation; dropout rate of 0.25
* 1 convolutional layer; 32 5x5 kernels; 2x2 stride; ReLU activation; dropout rate of 0.25
* 1 convolutional layer; 48 5x5 kernels; 2x2 stride; ReLU activation; dropout rate of 0.25
* 1 convolutional layer; 64 3x3 kernels; 1x1 stride; ReLU activation; no regularization
* 1 convolutional layer; 64 3x3 kernels; 1x1 stride; ReLU activation; no regularization
* 1 fully-connected layer; flattened dim of convolutional layers; ReLU activation; dropout rate of 0.5
* 1 fully-connected layer; 50 output dim; ReLU activation; dropout rate of 0.5
* 1 fully-connected layer; 10 output dim; ReLU activation; dropout rate of 0.5
* 1 fully-connected layer; 1 output dim; no activation; no regularization

In working through this, I tried a fairly tremendous number of variations on these building blocks, including simpler 3x3 conv stacks like my last project (which worked well, better actually from the outset, than the NVIDIA architecture, but I again fell back into wanting to successfully reproduce their results). I also tried deeper kernel sets for the conv layers, leaky activations, uniform stride, more and less extensive dropout regularization, l2 weight regularization, different color maps. In terms of data, I used differently calibrated augmentations (in fact, I ultimately left out the affine augmentation altogether, finding it finicky and not sufficiently necessary to warrant the time and effort), different ratios of augmented to honest samples, different splits of the training data, etc. Ultimately though, these changes usually led to overfitting or, in the case of certain examples, outright failure to learn. 

Training Regiment:
I used an Adam optimzer with the recommended 1E-4 rate, as it has worked time and time again for me, both in this class and in my own fumblings with text at work. I used MSE for my loss function, and it served at an adequate proxy, at least dynamically, for success on the track. As I struggled through my second week with the project, I couldn't help but think about writing a custom loss function that punished errors crossing the zero point more heavily, but fortunately (as who knows whether or not that would've worked) I got decent results before it came to that. I ran about 25k images through each epoch and ran five epochs, and found that anymore usually failed to improve at best, or overfit at worst. 

Final Thoughts:
Overall, I found this to be a tredmendously difficult project, if only because it was blisteringly empirical, and I seem to have chosen all of the worst initializations. That said, being forced to work through and consider every step (before ultimately realizing that one stupid bug had probably caused half of my issues) was very educational. Moreover, I suspect it gave a good feel for the combination experimental science/black magic that is most major machine learning initiatives. The first track being thoroughly conquered (save some smoothing on the straightaways) and the second being on its way (the net performed quite well on a number of chunks), I expect my interests in altering the model (use of VGG, different augmentations, combination compvis methods, etc) will probably wait until I get the GTA-V environment up and running on openAI's Universe, it being so much more realistic, and let's face it, potentially hilarious. Again though, this turned out to be one of my more frustrating, but ultimately rewarding projects to date.
