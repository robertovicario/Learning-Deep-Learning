### Summary of Lecture on Advanced Deep Learning Techniques

**Authors: Jean Hennebert and Martin Melchior**

#### Key Topics:
1. **Non-Sequential Architectures**
2. **Transfer Learning**
3. **Auto-Encoders**
4. **Self-Supervised Learning**
5. **Deep Learning (TSM-DeLearn)**

### Key Concepts Covered:

#### Principle of Data Generation:
- Generative systems create consistent new data from a seed, respecting learned temporal or spatial structures.
- Example: Text generation using a seed phrase.

#### Char-Level Many-to-One Approach with RNNs:
- Uses RNNs to predict the next token in a sequence.
- One-hot vectors represent characters.
- Softmax layer produces probability distributions, with outputs sampled rather than taken as argmax for variability.

#### Word Embedding:
- Projects words into a lower-dimensional vector space.
- Embeddings capture semantic similarities and differences between words.

#### Word2Vec Principles:
- Models producing word embeddings based on contextual similarity.
- Google’s Mikolov et al. (2013) papers are foundational.

#### Non-Sequential Architectures:
- **Keras Functional API**: Supports multiple paths, inputs, and outputs.
- **Example Models**:
  - MLP (Multilayer Perceptron)
  - CNN (Convolutional Neural Network)
  - CNNs with shared input layers and multiple feature extractors.

#### Transfer Learning:
- **Definition**: Re-using the feature extraction part of a pre-trained network.
- **Advantages**: Effective for tasks with limited labeled data.
- **Strategies**:
  1. Use pre-trained network as a feature extractor.
  2. Freeze initial layers, train new classification layers.
  3. Fine-tune the entire network with a small learning rate.
- **Applications**:
  - Image classification
  - Fine-grained recognition
  - Scene recognition

#### Auto-Encoders:
- **Definition**: Neural networks trained to reproduce their input, discovering efficient codings.
- **Applications**:
  - Data compression
  - Feature extraction
  - Denoising
  - Image in-painting
  - Network initialization
  - Anomaly detection

#### Self-Supervised Learning (SSL):
- **Definition**: Learning from unlabeled data using pretext tasks.
- **Example**: BERT uses masked word prediction and next sentence prediction.
- **Framework**: SimCLR (contrastive learning approach) by Google Research.

### Practical Implementations:
- **Keras and Transfer Learning**:
  - Examples using MobileNetV2 and CIFAR10 dataset.
  - Implementation involves saving images, using `ImageDataGenerator`, and fine-tuning pre-trained models.

### Good Practices in Model Training:
- **Callbacks in Keras**: Functions applied during training to monitor and save models.
- **Consistent Variable Naming**: For clarity and ease of connecting model components.
- **Reviewing Layer Summaries**: Ensuring model connections are as expected.

### Real-Life Examples and Applications:
- **MeteoCam**: Road state recognition.
- **NSFW Detector**: Classification of safe vs. NSFW images.

The lecture comprehensively covers advanced topics in deep learning, emphasizing practical implementation, good practices, and the utility of transfer learning and self-supervised learning in various applications.

### Summary of Lecture on Advanced Generative Models

**Authors: Jean Hennebert and Martin Melchior**

#### Key Topics:
1. **Generative Models**
2. **Variational Auto-Encoder (VAE)**
3. **Generative Adversarial Network (GAN)**
4. **Diffusion Models**

### Key Concepts Covered:

#### Introduction:
- **Generative Models**: Learn the distribution underlying the training data to generate new samples.
- **Possible Goals**:
  - Data augmentation
  - Simulating real-world scenarios
  - Unsupervised learning of representations
  - Realistic samples for artwork

#### Auto-Encoder:
- **Architecture**: Neural network trained to reproduce its input \(x\). No labels needed - unsupervised training. 
- **Latent Space**: Compressed low-dimensional representation of input data.
- **Applications**:
  - Data compression
  - Feature extraction for downstream tasks
  - Denoising
  - Image in-painting
  - Historically used for layer-wise training of deep neural networks

#### Variational Auto-Encoder (VAE):
- **Goal**: Generate samples following the distribution underlying the original data.
- **Latent Space Sampling**: Use an encoder to map inputs to a latent space and a decoder to generate outputs from this space.
- **KL-Divergence**: Minimizing the Kullback-Leibler divergence between the learned distribution and the true data distribution.

#### Generative Adversarial Networks (GANs):
- **Main Idea**: Train two networks (Generator and Discriminator) in an adversarial setting.
  - **Generator**: Generates new samples.
  - **Discriminator**: Judges whether samples are real or fake.
- **Loss Functions**:
  - **Discriminator Loss**: Measures how well the discriminator distinguishes real from fake samples.
  - **Generator Loss**: Measures how well the generator fools the discriminator.
- **DCGAN**: Deep Convolutional GAN, using convolutional layers for better performance.

#### Diffusion Models:
- **Goal**: Learn to generate images by adding noise to data and then learning to remove it.
- **Forward Diffusion Process**: Adds noise to the data in multiple steps.
- **Reverse Diffusion Process**: Neural network learns to denoise the data, reversing the diffusion process.
- **Optimization**: Minimize negative log-likelihood or variational upper bound.
- **Applications**: High-resolution image synthesis, conditional generation, and text-to-image models.

#### Important Techniques and Papers:
- **Principal Component Analysis (PCA)**
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
- **Uniform Manifold Approximation and Projection (UMAP)**
- **Notable Papers**:
  - "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (Sohl-Dickstein et al, 2015)
  - "Denoising Diffusion Probabilistic Models" (Ho et al, 2020)
  - "Diffusion Models Beat GANs on Image Synthesis" (Dhariwal et al, 2021)

#### Further Reading and Resources:
- Blog articles and tutorials on VAEs, GANs, and Diffusion Models.
- Detailed implementation guidelines and mathematical derivations for these models.

The lecture provides a comprehensive overview of advanced generative models, including theoretical foundations, practical implementations, and applications in various domains.

### Summary of Lecture on Advanced Architectures and Training Strategies: Transformers

**Authors: Jean Hennebert and Martin Melchior**

#### Key Topics:
1. **Recaps on Seq2Seq and Attention**
2. **Transformer Architecture**
3. **Self-Attention Layer**

### Key Concepts Covered:

#### Recaps:
- **Sequence to Sequence Model (Seq2Seq)**:
  - Maps sequences of different lengths, requiring the entire input sequence to start predicting the target sequence.
  - **Encoder**: Processes input sequence, compresses it into a context vector (sentence embedding).
  - **Decoder**: Initialized with the context vector to generate the output.
  - Initial architectures were based on RNNs (Recurrent Neural Networks), which struggled with long sequences. Attention mechanisms were introduced to address this.

- **Applications in NLP**:
  - Word embedding combined with RNNs for tasks like named entity recognition and sentiment analysis.
  - Seq2Seq models enabled machine translation, chatbots, and other applications.

- **Attention Mechanisms**:
  - Interface between encoder and decoder, providing the decoder with information from all or parts of the encoder hidden states.
  - Types: Global (uses all encoder hidden states) and Local (uses a subset).

#### Transformer Architecture:
- **Definition**: Transformers handle sequential data without requiring ordered processing, relying on attention mechanisms.
- **Advantages over RNNs**:
  - Do not need to process data in order, allowing for better parallelization.
  - Can capture context more effectively with attention mechanisms.

- **Key Paper**: "Attention is All You Need" by Vaswani et al. (2017), which demonstrated that attention mechanisms alone, without RNNs, could achieve significant performance gains.

- **High-Level Architecture**:
  - Composed of an encoding part (stack of encoder modules) and a decoding part (stack of decoder modules).
  - Encoder and decoder modules are identical in structure but have different inner configurations.
  - **Self-Attention Layer**: Enhances word representations by considering the context provided by other words in the sequence.

- **Applications in NLP**:
  - Transformers have become the standard for NLP tasks, surpassing LSTM-based architectures.
  - Notable models: BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer).

#### Self-Attention Layer:
- **Principles**:
  - Allows the model to look at positions in the input sequence for better word representation.
  - Implements an attention mechanism by looking at adjacent inputs, providing an enhanced representation of the current input.

- **Implementation**:
  - Based on key-value-query computation, where inputs are mapped to key, value, and query vectors using dense layers.
  - Attention scores are computed, normalized with softmax, and used to weight the value vectors.

- **Extensions**:
  - Include bias terms, positional encoding, residuals, and normalization.
  - Multi-head attention and layer stacking for enhanced performance.

#### Further Resources:
- **Videos and Articles**: Various resources for in-depth understanding of transformers, attention mechanisms, and their applications in NLP and beyond.

This lecture provides a comprehensive overview of transformer architectures, their advantages over traditional RNNs, and their applications in natural language processing and other domains.

### Summary of Lecture on Deep Learning Frameworks, Hardware, and Impact

**Authors: Jean Hennebert and Martin Melchior**

#### Key Topics:
1. **Recaps from Previous Lectures**
2. **Computational Graphs and Implementations**
3. **Deep Learning Frameworks**
4. **CPU vs. GPU**
5. **AI and Sustainability**

### Key Concepts Covered:

#### Recaps from Previous Lectures:
- **Activation Functions**:
  - Non-squeezing activation functions like ReLU and Leaky ReLU are preferred.
- **Parameter Initialization**:
  - Stabilize the distribution of gradients and activations at the start of training.
- **Batch Normalization**:
  - Normalize activations to improve training stability and speed.

#### Computational Graphs:
- **Advantages**:
  - Nodes represent operations or variables, and edges represent links between nodes.
  - Useful for defining complex operations and allowing various optimization techniques.
- **Backpropagation and Chain Rule**:
  - Calculate gradients of the output concerning any parameter using the chain rule.

#### Deep Learning Frameworks:
- **Popular Frameworks**:
  - Academia: Theano, Torch, Caffe, MXNet.
  - Companies: TensorFlow (Google), PyTorch (Facebook), Caffe2 (Facebook), MXNet (Amazon), CNTK (Microsoft), PaddlePaddle (Baidu).
- **TensorFlow vs. PyTorch**:
  - TensorFlow: Based on Theano, initially static graph execution, now supports eager mode.
  - PyTorch: Based on Torch, dynamic eager execution, more Pythonic, preferred for rapid prototyping.
- **Framework Advantages**:
  - Easy construction of computational graphs.
  - Simplifies computation of losses and gradients.
  - Includes state-of-the-art regularization and optimization strategies.
  - Easy switching between CPU and GPU.

#### CPU vs. GPU:
- **CPU (Central Processing Unit)**:
  - Few cores (~10), fast (~4 GHz), lots of cache, great for sequential tasks.
- **GPU (Graphical Processing Unit)**:
  - Many cores (~1,000), slower (~1.5 GHz), fewer caches, great for parallel tasks.
  - Used extensively in deep learning for tasks like matrix multiplication.

#### Programming GPUs:
- **CUDA**:
  - Low-level API for programming NVIDIA GPUs.
  - Higher-level APIs: cuBLAS, cuFFT, cuDNN.
- **OpenCL**:
  - Similar to CUDA, runs on various hardware, usually slower on NVIDIA GPUs.
- **HIP**:
  - Write once in HIP C++ and port to NVIDIA and AMD platforms.

#### AI and Sustainability:
- **AI for Sustainability**:
  - Digital platforms where AI solutions advance sustainable development goals.
  - Applications in various domains such as environmental monitoring, energy optimization, etc.
- **Sustainability of AI**:
  - Evaluating and reducing the environmental impact of AI models.
  - Optimizing models to reduce energy consumption and carbon footprint.
  - Ensuring ethical AI applications.

#### Measuring AI's Environmental Impact:
- **Factors to Consider**:
  - Electrical energy consumed, carbon intensity, and total carbon emissions.
  - Power Usage Effectiveness (PUE) and Pragmatic Scaling Factor (PSF).

#### Tools and APIs:
- **Electricity Maps**:
  - Provides data on the electricity consumption and carbon footprint.
- **CO2 Signals**:
  - API for accessing CO2 data, used by tools like Code Carbon.
- **Other Resources**:
  - Our World In Data, Low Carbon Power.

### Conclusion:
This lecture provided a comprehensive overview of deep learning frameworks, their advantages, and implementation strategies. It also discussed the importance of hardware (CPU vs. GPU) in deep learning, programming GPUs, and the sustainability of AI, emphasizing the need to measure and reduce the environmental impact of AI technologies.
