# LAB - Generative AI: Text to Image

![](https://user-images.githubusercontent.com/970858/63474771-d6734700-c469-11e9-83bb-9429da563909.png)

## Introduction

In this lab, we will explore various generative AI models to generate, modify, and enhance images. We will use models such as **Diffusion Models**, **GANs**, and **VAEs** to apply different transformations to images. The exercises are structured in a step-by-step manner, designed to be completed on **Google Colab** for optimal GPU usage.

Ensure you have access to a GPU by going to `Runtime > Change runtime type > GPU` before running the code.

## Setup Instructions

To run the notebook in Google Colab and reproduce the results, follow these steps:

1. **Upload the notebook to Google Colab**:
   - Open [Google Colab](https://colab.research.google.com/).
   - Select **File > Upload Notebook**.
   - Upload the provided `.ipynb` notebook.

2. **Install dependencies**:
    Add the following cell at the beginning of the notebook and run it to install the necessary dependencies:

    ```python
    !pip install diffusers transformers torch torchvision tensorflow scipy
    ```

3. **Start working on the notebook**:
   After the libraries have been installed, proceed to run each cell in the notebook to complete the exercises.

 # Workflow
 
 In this lab, we will explore various generative AI models to generate, modify, and enhance images. We will use models such as **Diffusion Models**, **GANs**, and **VAEs** to apply different transformations to images. The exercises are structured in a step-by-step manner, designed to be completed on **Google Colab** for optimal GPU usage.
 
 Ensure you have access to a GPU by going to `Runtime > Change runtime type > GPU` before running the code. Each iteration focuses on a specific task, and by following the steps, you will better understand how generative models work.


---

## Setup
 
 First of all, make sure you are `using a GPU` and connect to your `Drive Account` to save your generated images.

---

## Iteration 1: Generate Images from Text using Stable Diffusion Model
 
 In this task, you will provide a textual description, and the model will generate an image that matches the description.
 
 


---

### Step 1: Install Dependencies
 
 In this first step, students need to install all the required dependencies to work with the Stable Diffusion model, image processing libraries, and the Pytorch and Tensorflow frameworks.
 
 - `diffusers`: A library to work with diffusion models such as Stable Diffusion.
 - `transformers`: To handle text models.
 - `torch`: Pytorch, used for neural network models.
 - `torchvision`: For image transformations.
 - `tensorflow`: To load the pre-trained InceptionV3 model.
 - `scipy`: A scientific computing library used for calculating entropy.
 


---

### Step 2: Load Stable Diffusion Pre-trained Model
 
 Load a pre-trained `"CompVis/stable-diffusion-v1-4"` Stable Diffusion model.


---

### Step 3: Generate Images from Text Prompts
 
 In this step, we will write a function to generate images from a list of text prompts. The function will iterate over the list of prompts, generate images, and return them as a list.
 The images must be saved as `.png` in the `it1_generated_images` folder.
 
 - **Name:** `generate_images`
 - **Parameters:**
     - `prompts` (list): List of text descriptions.
     - `num_images` (int): Number of images to generate per prompt (default is 1).
 - **Returns:** A list of generated images.
 
 > 🔝 Don't forget to save the images!

---


 Now define 2 prompts saved in a list and call the `generate_images()`function to create the images. This function receives the following parameters:
 
   - prompts (list)
   - num_images=1
 


---

### Step 4: Visualize Generated Images
 
 create a function to visualize the generated images along with their respective text prompts.
 
 - **Function:** `visualize_images_with_prompts`
 - **Parameters:**
     - `images` (list): List of generated images.
     - `prompts` (list): List of corresponding text prompts.
     - `num_images_per_prompt` (int): Number of images to visualize per prompt.
 - **Returns:** A visual grid of images with their prompts.


---

## Iteration 2: Add Color to a Black-and-White Image using Diffusion Models
 
 In this iteration, you will take a black-and-white image and apply a Diffusion Model to colorize it.
 
 ![](https://res.cloudinary.com/dbjlaop5w/image/upload/v1729177524/codeskills/generative-ai/Text-to-Image%20Models%20for%20Content%20Creation/wfwgnwjbf0c14njivn2u.png)
 
 


---

### Step 1: Import needed dependencies


---

### Step 2: Load the Grayscale Image
 In this step, you'll load the `grayscale_image.jpg` located in this repo. You will use the **PIL** library to open the image and display it using **matplotlib**.

---

### Step 3: Prepare the Environment
 
 Now, let's clone the [`DeOldify`](https://github.com/jantic/DeOldify.git) repository, which contains the colorization model.
 
 > 💡Check the files in the GitHub Repository: https://github.com/jantic/DeOldify.git
 
 There's a `requirements-colab.txt` file. Use it to install the required dependencies.

---

### Step 4: Download the Model's weights for Colorization
 
 In this step, you'll download the pre-trained model weights necessary for colorization.
 
 - **Download the model**: `https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth`
 - **Move the model**: make sure to locate the model (`ColorizeArtistic_gen.pt`) in the `/models/` folder.
 
 >  💡Check `wget` command.

---

### Step 5: Run the Colorization Model
 
 Now that the model is set up, you can load your grayscale image into the model and run the colorization process.
 
 1. Save the image path in the `image_path` variable.
 2. Use the `colorizer.plot_transformed_image()` method to colorize the image. Use the following values for each param:
   - `path=image_path`
   - `render_factor=35`
   - `figsize=(8,8)`

---

## Iteration 3: Improve Image Quality using GANs
 
 In this task, you will improve the quality of a low-resolution image using **Stable Diffusion**. You will apply the reverse diffusion process to denoise the image and enhance its resolution.
 
 ![](https://res.cloudinary.com/dbjlaop5w/image/upload/v1729203799/codeskills/advanced-python/Object%20Oriented%20Programming%20in%20Python/hfpwuwampxirlappg5fp.png)


---


 ### Step 1: Clone the Real-ESRGAN Repository and Install Dependencies
 First, we will clone the [Real-ESRGAN repository](https://github.com/xinntao/Real-ESRGAN.git) and install the required dependencies for running the model.
 
 1. **Clone the repository**
 ```bash
 https://github.com/xinntao/Real-ESRGAN.git
 ```


---

2. **Update `requirements.txt`**
 
 > ⚠️ WARNING!
 the `torchvision.transforms.functional_tensor` module no longer exists in the current version of `torchvision`. To avoid errors, access to the `requirements.txt` file before installing dependencies and write:
 ```bash
 torchvision==0.12.0
 ```
 instead of:
 ```bash
 torchvision
 ```
 
 3. **Install dependencies**


---

### Step 2: Download the Pre-trained Model Weights
 
 Next, you will download the pre-trained model weights for Real-ESRGAN_x4plus, which will be used for enhancing image resolution.
 
 - **Download the weights**: `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth`
 - **Move the weights**: locate the `RealESRGAN_x4plus.pth` file in the `/weights/` folder.

---

### Step 3: Create the output folder
 
 You will now prepare the environment to create the images.
 
 Create the `improved_images` folder if it does not exist yet.


---

### Step 4: Run Inference to Enhance Images
 
 Now, you will run the Real-ESRGAN model on the low quality images to enhance their resolution.
 
 You can run it with the following command:
 
 ```bash
 !python <path_to_inference_realesrgan.py> -n <model_name> -i <input_directory> -o <output_directory> --outscale <scale_factor> [--face_enhance]
 ```
 
 - `<model_name>`: Specifies the model to be used (e.g., RealESRGAN_x4plus).
 - `<input_directory>`: Path to the directory containing the input (low-quality) images.
 - `<output_directory>`: Path where the enhanced images will be saved.
 - `--outscale <scale_factor>`: Defines the scale factor for upscaling the images (e.g., 3.5).
 - `--face_enhance` (optional): Enhances facial details in the images, useful when images contain faces.
 
 > ⚠️  WARNING!
 
 > To avoid the `ModuleNotFoundError` when the script tried to import a missing version.py, we must create a placeholder file with basic version information. Run the following command before running the inference.

---

### Step 5: Visualize the Results
 
 Finally, you can visualize the results by comparing the input images with the enhanced output images.

---

### Bonus 1: Improving the Prompt for Stable Diffusion with Ollama
 
 In this bonus task, the goal is to use a language model from **Ollama** to generate or improve the text prompts that will be passed to the **Stable Diffusion** model. This can help you create more detailed and rich descriptions, improving the quality of the generated images.
 
 #### Guidelines
 1. **Explore Ollama**: Research how to use **Ollama** to access language models that can help you generate or refine prompts.
 
 2. **Prompt generation**: Use Ollama’s language model to transform a basic input (e.g., "A mountain landscape") into a more detailed description.
 
 3. **Integration with Stable Diffusion**: Once you have the refined prompt, use it with the **Stable Diffusion** model to generate the image.
 
 4. **Performance comparison**: Compare the results of image generation (using Ollama or not).


---

### Bonus 2: Age a Person's Face using VAEs
 
 In this iteration, you will use a VAE to transform a person's face from a young version to an older version.
 
 #### Guidelines:
 
 1. Load a dataset of young and old faces to train a VAE.
 2. Encode the young face into the latent space.
 3. Manipulate the latent vector to introduce aging features.
 4. Decode the latent vector to generate the aged face.
 5. Visualize and compare the results.
