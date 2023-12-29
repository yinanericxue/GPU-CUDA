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



