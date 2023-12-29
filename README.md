# GPU and CUDA

#### https://codeconfessions.substack.com/p/gpu-computing
#### CPU (aka host): handle complex tasks but the concurrency is limited, only support a few cores.
#### GPU (aka device): handle simple tasks but support thousands of cores, as it was originally designed to support computer graphics and video, but now it is used to accelerate calculations.
#### GPU has much faster throughput (the higher the better) and lower latency (the lower the better).
#### The bridge (PCI-Express) between them is very slow.

![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/684e91e2-806a-4bdb-901f-d2daff309bb8)
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/63cbab49-66ed-4c11-ae85-2b74791c8965)

### Rendering
#### https://en.wikipedia.org/wiki/Rendering_(computer_graphics)#:~:text=Rendering
#### https://www.heavy.ai/technical-glossary/gpu-rendering

#### OpenGL for Games, C Programming API, supported by all GPU vendors - Intel, AMD, NVIDIA, ARM，Huawei
https://en.wikipedia.org/wiki/List_of_OpenGL_applications

### CUDA for AI/ML, only NVIDIA  ( Graphics Card -> Bitcoin -> AI/ML)
#### TensorFlow over cnDNN over CUDA over GPU Hardware
#### PyTorch over cnDNN over CUDA over GPU Hardware

### PC Architecture
#### https://arstechnica.com/features/2004/07/pcie/
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/7018e664-19fd-428a-b9f2-5d83ad1a401d)

#### For example, if there are two numbers in the memory (a and b), they must be loaded into the CPU registers for the flops operations to be performed, then they are loaded back into the memory.
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/d60a24ae-4b9c-47d7-b1bb-e1edcc66c44e)
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/32d76f6a-4947-4613-a641-6d138951eca7)

### Core Logic Chipset
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/abe767d0-af6e-49f0-a03e-209d6d44be7d)

### CPU and GPU System Architecture
#### https://www.intechopen.com/chapters/54968
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/78fad729-ade1-40a8-a886-0023136d8694)
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/698fb803-e2bd-4fb6-9214-0fc837487065)

### Distributed Training: Model Parallelism vs Data Parallelism
#### https://neptune.ai/blog/distributed-training

### NVLINK vs. NVSWITCH
#### https://www.anandtech.com/show/12581/nvidia-develops-nvlink-switch-nvswitch-18-ports-for-dgx2-more
#### https://www.nvidia.com/en-us/data-center/nvlink/
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/23f2a256-2106-4f34-9f31-d10a0126883d)

### CUDA Programming
#### https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/46987d9e-d0b1-446f-9bc3-5d7f7952467e)
#### http://w-uh.com/posts/230122-CUDA_coding.html
<img width="404" alt="Screen Shot 2023-12-28 at 4 54 29 PM" src="https://github.com/yinanericxue/GPU-CUDA/assets/102645083/a0abc513-01f0-4a61-ab02-3e38ee87fe35">
<img width="400" alt="image" src="https://github.com/yinanericxue/GPU-CUDA/assets/102645083/1a664550-3ca0-4d4a-b2f7-db093091ef81">

### NVCC (CUDA Toolkit)
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/c4681f77-1374-4462-a476-38107e8f3c16)

### CPU Process Context: Linux and Web/Parallel Programming
#### https://www.baeldung.com/cs/os-cpu-context-switch
#### https://www.geeksforgeeks.org/context-switch-in-operating-system/
#### Multiple Processes share the same CPU.
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/33418023-e3ac-4f60-8285-c2aadd79b7d0)

### GPU CUDA Context and its VRAM usage
#### https://www.tutorialspoint.com/is-it-possible-to-share-a-cuda-context-between-applications
#### https://stackoverflow.com/questions/43244645/what-is-a-cuda-context
#### https://docs.nvidia.com/deploy/mps/index.html
#### Multiple Processes share the same GPU.
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/2ea5f21e-fd19-4165-b617-368f90cbd7d6)
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/47866405-7c59-498a-9602-d636fa62fdd6)

### Lazy initialization of CUDA context
#### https://stackoverflow.com/questions/7534892/cuda-context-creation-and-resource-association-in-runtime-api-applications![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/c54c0a51-fb45-4417-a703-ee37586ccb5b)
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/a966d3f4-fb50-4cbe-87b1-86ecd1cbb64c)

### VRAM usage of CUDA Context
#### https://discuss.pytorch.org/t/what-is-the-initial-1-3gb-allocated-vram-when-first-using-cuda/122079
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/af0b947a-073d-4265-b950-595d6a80d029)

#### Actual GPU memory is 1.3/4 GB, and the extra memory are used by CUDA context (similar to traditional process stack pointer)
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/192017c1-7e2c-4707-a1a5-7561dd2bb08f)
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/a6e4ec42-7826-4ba9-bf90-e1f27de675b7)

### Basic PyTorch model, involving moving information to GPU
<img width="680" alt="Screen Shot 2023-12-28 at 5 51 53 PM" src="https://github.com/yinanericxue/GPU-CUDA/assets/102645083/83161edc-a35d-4936-b3f2-ffd13478f9f4">

### Basic linear model with Torch computed with GPU
<img width="612" alt="Screen Shot 2023-12-28 at 5 53 00 PM" src="https://github.com/yinanericxue/GPU-CUDA/assets/102645083/a818337b-4a06-4b8d-95ff-ff095cb84a31">

### CUDA Stream
https://leimao.github.io/blog/CUDA-Stream/#CUDA%20Stream
https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/46c1580f-d02b-49c9-907b-ee0484e4e1b7)

##### Async and Sync Operations
https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
https://www.cs.emory.edu/~cheung/Courses/355/Syllabus/94-CUDA/SLIDES/s06.html

![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/84895566-1f46-4871-baf3-68af955e5f03)
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/c78cd301-8b7a-449c-8b93-cc702208a423)

### Full Software Stack

https://www.tensorflow.org/install/gpu
https://pytorch.org/get-started/locally/

#### Python apps / PyTorch or TensorFlow / cuDNN / CUDA Runtime / CUDA Driver / GPU Driver

#### cuda toolkits include cuda runtime, nvcc, example code (helloworld)

<img width="320" alt="image" src="https://github.com/yinanericxue/GPU-CUDA/assets/102645083/92637a40-cc4e-4843-93f5-9d3fa9c18df9">

### cuDNN
#### https://developer.nvidia.com/cudnn
#### https://blog.roboflow.com/what-is-cudnn/
#### https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
![image](https://github.com/yinanericxue/GPU-CUDA/assets/102645083/d79d91c2-f9a7-4f2c-82b9-767f293312c4)
