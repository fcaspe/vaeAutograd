
# Implements auto-encoding variational Bayes.

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
    
from autograd import grad
from data import load_mnist
from data import save_images as s_images
from autograd.misc import flatten # This is used to flatten the params (transforms a list into a numpy array)

# images is an array with one row per image, file_name is the png file on which to save the images

def save_images(images, file_name): return s_images(images, file_name, vmin = 0.0, vmax = 1.0)

# Sigmoid activiation function to estimate probabilities

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Relu activation function for non-linearity

def relu(x):    return np.maximum(0, x)

# This function intializes the parameters of a deep neural network

def init_net_params(layer_sizes, scale = 1e-2):

    """Build a (weights, biases) tuples for all layers."""

    return [(scale * npr.randn(m, n),   # weight matrix
             scale * npr.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

# This will be used to normalize the activations of the NN

# This computes the output of a deep neuralnetwork with params a list with pairs of weights and biases

def neural_net_predict(params, inputs):

    """Params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       Applies batch normalization to every layer but the last."""

    for W, b in params[:-1]:
        outputs = np.dot(inputs, W) + b  # linear transformation
        inputs = relu(outputs)         # nonlinear transformation

    # Last layer is linear

    outW, outb = params[-1]
    outputs = np.dot(inputs, outW) + outb

    return outputs

# This implements the reparametrization trick

def sample_latent_variables_from_posterior(encoder_output):

    # Params of a diagonal Gaussian. 
    # I think The encoder network generates a vector of 2 values per latent variable: a mean and a log_std, parameters of a gaussian.
    # Do we have then an encoder output of 100 values. Yes! (50 means 50 stds).

    D = np.shape(encoder_output)[-1] // 2
    mean, log_std = encoder_output[:, :D], encoder_output[:, D:]

    # print("[DEBUG] sample_latent_variables...() encoder_output shape {}".format(np.shape(encoder_output)))
    # print("[DEBUG] sample_latent_variables...() mean shape {} log_std shape {}".format(np.shape(mean),np.shape(log_std)))
    # print("[DEBUG] D: {}".format(D))
    
    # TODO use the reparametrization trick to generate one sample from q(z|x) per each batch datapoint
    # use npr.randn for that. (same as np.random.normal(0,1)
    # The output of this function is a matrix of size the batch x the number of latent dimensions
    
    #python breakpoint
    #import pdb; pdb.set_trace()
    log_var = log_std*2
    batch_size = np.shape(encoder_output)[0]
    z = np.zeros((batch_size,D))
    var = np.exp(log_var)
    z = mean + np.sqrt(var)*npr.randn(batch_size,D)
    return z

# This evlauates the log of the term that depends on the data

def bernoulli_log_prob(targets, logits):

    # logits are in R
    # Targets must be between 0 and 1

    # TODO compute the log probability of the targets given the generator output specified in logits
    # sum the probabilities across the dimensions of each image in the batch. The output of this function 
    # should be a vector of size the batch size
    
    # So we have a binary overlay (the targets) and logits that explain on each pixel 
    # The probability of that pixel being 1.
    # In order to define how well a target has been explained by the logits, we compute an overall probability
    # Per target, by summing the log prob of each black pixel being black and each white pixel being white.
    # Ideally, if the logits (output of decoder) perfectly generates the target, the log prob gets to 0 (probability of 1).
    # NOTE: Targets and logits are flattened
    
    # print("[DEBUG] bernoulli_log_prob() target shape: {} logit shape: {}".format(targets.shape,logits.shape))
    batch_size = np.shape(targets)[0]
    # print("[DEBUG] bernoulli_log_prob() Batch size is: {}".format(batch_size))
    log_probs = np.zeros(batch_size)

    prob_nn_out = sigmoid(logits)
    logprob_map = np.log(targets * prob_nn_out + (1 - targets)*(1 - prob_nn_out))
    log_probs = np.sum(logprob_map,axis = 1)
    # print("[DEBUG] log_probs shape {}".format(log_probs.shape))
    return log_probs

# This evaluates the KL between q and the prior

def compute_KL(q_means_and_log_stds):
    
    D = np.shape(q_means_and_log_stds)[-1] // 2
    mean, log_std = q_means_and_log_stds[:, :D], q_means_and_log_stds[:, D:]

    # TODO compute the KL divergence between q(z|x) and the prior (use a standard Gaussian for the prior)
    # Use the fact that the KL divervence is the sum of KL divergence of the marginals if q and p factorize
    # The output of this function should be a vector of size the batch size
    
    # Equation 12
    # Compute how well the q distribution (50 means and 50 log_std, per instance)
    # fit the prior distribution
    
    batch_size = np.shape(q_means_and_log_stds)[0]
    KL = np.zeros(batch_size)
    log_var = log_std*2
    # print("[DEBUG] compute_KL() Batch size is: {}".format(batch_size))

    KL = 0.5* np.sum(np.exp(log_var) + (mean**2 - 1 - log_var),axis = 1) #sum across all latent variables.
    # print("[DEBUG] compute_KL() KL shape is: {}".format(KL.shape))
    return KL

# This evaluates the lower bound

def vae_lower_bound(gen_params, rec_params, data):

    # TODO compute a noisy estiamte of the lower bound by using a single Monte Carlo sample:

    # 1 - compute the encoder output using neural_net_predict given the data and rec_params
    # 2 - sample the latent variables associated to the batch in data 
    #     (use sample_latent_variables_from_posterior and the encoder output)
    # 3 - use the sampled latent variables to reconstruct the image and to compute the log_prob of the actual data
    #     (use neural_net_predict for that)
    # 4 - compute the KL divergence between q(z|x) and the prior (use compute_KL for that)
    # 5 - return an average estimate (per batch point) of the lower bound by substracting the KL to the data dependent term

    #1. Encoder = recognition network = p(z/x) that we transform in -> q(z/x).
    encoder_output = neural_net_predict(rec_params,data)
    
    #2. The encoder provides N= [2x(# LATENT VARIABLES)] parameters per instance: a mean and a std per latent variable. 
    #   That is: the output distribution factorizes across multiple gaussians. (# gaussians = # LATENT VARS)
    #   From each gaussian of each instance we will sample a single value using the reparametrization trick. Length vector = # LATENT VARS.
    
    sampled_latent_vars = sample_latent_variables_from_posterior(encoder_output)
    
    #3. Generate the logit matrix using the decoder network = generator network = p(x/z).
    #   Then compute the logprob for every instance.
    decoder_output = neural_net_predict(gen_params,sampled_latent_vars)
    decoder_logprobs = bernoulli_log_prob(data,decoder_output)
    
    # 4. KL DIVERGENCE. Since we assume our prior is also gaussian, as our multimodal approx q(z/x), the distr has a closed form.
    KL = compute_KL(encoder_output)
    
    # 5. Return the average of the difference between the lower bound and the KL for all instances in batch.
    return np.mean(decoder_logprobs - KL)


if __name__ == '__main__':

    # Model hyper-parameters

    npr.seed(0) # We fix the random seed for reproducibility

    latent_dim = 50
    data_dim = 784  # How many pixels in each image (28x28).
    n_units = 200
    n_layers = 2

    gen_layer_sizes = [ latent_dim ] + [ n_units for i in range(n_layers) ] + [ data_dim ]
    rec_layer_sizes = [ data_dim ]  + [ n_units for i in range(n_layers) ] + [ latent_dim * 2 ]

    # Training parameters

    batch_size = 200
    num_epochs = 30
    learning_rate = 0.001

    print("Loading training data...")

    N, train_images, _, test_images, _ = load_mnist()

    # Parameters for the generator network p(x|z)

    init_gen_params = init_net_params(gen_layer_sizes)

    # Parameters for the recognition network p(z|x)

    init_rec_params = init_net_params(rec_layer_sizes)

    combined_params_init = (init_gen_params, init_rec_params) 

    num_batches = int(np.ceil(len(train_images) / batch_size))

    # We flatten the parameters (transform the lists or tupples into numpy arrays)

    flattened_combined_params_init, unflat_params = flatten(combined_params_init)

    # Actual objective to optimize that receives flattened params

    def objective(flattened_combined_params):

        combined_params = unflat_params(flattened_combined_params)
        data_idx = batch
        gen_params, rec_params = combined_params

        # We binarize the data

        on = train_images[ data_idx ,: ] > npr.uniform(size = train_images[ data_idx ,: ].shape)
        images = train_images[ data_idx, : ] * 0.0
        images[ on ] = 1.0

        return vae_lower_bound(gen_params, rec_params, images) 

    # Get gradients of objective using autograd.

    objective_grad = grad(objective)
    flattened_current_params = flattened_combined_params_init

    # ADAM parameters
    # TODO write here the initial values for the ADAM parameters (including the m and v vectors)
    # you can use np.zeros_like(flattened_current_params) to initialize m and v
    t = 1
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # m and v will be used to optimize the params, in a flattened form.
    m = np.zeros_like(flattened_current_params)
    v = np.zeros_like(flattened_current_params)


    # We do the actual training

    for epoch in range(num_epochs):

        elbo_est = 0.0

        for n_batch in range(int(np.ceil(N / batch_size))):

            batch = np.arange(batch_size * n_batch, np.minimum(N, batch_size * (n_batch + 1)))
            # Obtain gradient by computing a batch

            # DOUBT: How does the objective() function sees the 'batch' variable?
            grad = objective_grad(flattened_current_params)

            # TODO Use the estimated noisy gradient in grad to update the paramters using the ADAM updates

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad*grad
            m_corrected = m / ( 1 - (beta1 ** t))
            v_corrected = v / ( 1 - (beta2 ** t))
            # Update current params using the momentum and lr (alpha). (since we try to maximize the lower bound, 
            # we compute: params = params + update.
            updating_term = alpha * m_corrected/(np.sqrt(v_corrected) + epsilon)
            flattened_current_params = flattened_current_params + updating_term

            elbo_est += objective(flattened_current_params)

            t += 1

        print("Epoch: %d ELBO: %e" % (epoch, elbo_est / np.ceil(N / batch_size)))

    # We obtain the final trained parameters

    gen_params, rec_params = unflat_params(flattened_current_params)

    # TODO Generate 25 images from prior (use neural_net_predict) and save them using save_images
    priors = npr.randn(25,50)
    logits = neural_net_predict(gen_params,priors)
    images = sigmoid(logits)
    save_images(images,'images_from_random_prior.png')
    
    # TODO Generate image reconstructions for the first 10 test images (use neural_net_predict for each model) 
    # and save them alongside with the original image using save_images

    test_samples = test_images[:10,:]
    posterior = neural_net_predict(rec_params,test_samples)
    #encoded_test_samples = sample_latent_variables_from_posterior(posterior)
    encoded_test_samples = posterior[:,:50] #We use the mean of the factor distributions as the latent variables. No sampling.
    logits = neural_net_predict(gen_params,encoded_test_samples)
    decoded_test_samples = sigmoid(logits)

    images = np.append(test_samples,decoded_test_samples,axis=0)
    save_images(images,'images_from_encoder.png')
    
    num_interpolations = 25
    n_image = 0
    for i in range(5):

        # TODO Generate 5 interpolations from the first test image to the second test image, 
        # for the third to the fourth and so on until 5 interpolations
        # are computed in latent space and save them using save images. 
        # Use a different file name to store the images of each iterpolation.
        # To interpolate from  image I to image G use a convex conbination. Namely,
        # I * s + (1-s) * G where s is a sequence of numbers from 0 to 1 obtained by numpy.linspace
        # Use mean of the recognition model as the latent representation.

        s = np.linspace(0,1,num_interpolations)
        s = s.reshape((num_interpolations,1))

        I = encoded_test_samples[n_image,:]
        I = I.reshape((1,50))
        n_image += 1

        G = encoded_test_samples[n_image,:]
        G = G.reshape((1,50))
        n_image += 1
        interpolated_latent_vars = ( s @ G + (1 - s) @ I )
        logits = neural_net_predict(gen_params,interpolated_latent_vars)
        interpolated_images = sigmoid(logits)
        save_images(interpolated_images,"interpolated_set_{}.png".format(i))

