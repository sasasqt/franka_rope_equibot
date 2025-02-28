require e3nn `pip install e3nn` and dgl `https://www.dgl.ai/pages/start.html`


```
pip install torch-scatter
pip install torch-cluster
pip install transforms3d
pip install potpourri3d
```

FOR WINDOWS
install visual studio 2022 with desktop dev with c++

replace `Visual Studio 16 2019` with `Visual Studio 17 2022`

```
cd dgl
md build
cd build

cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" -DDMLC_FORCE_SHARED_CRT=ON .. -G "Visual Studio 17 2022"

"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\msbuild" dgl.sln /m

CD ..\python
python setup.py install
```










cuda version install:

```
conda install nvidia::libcurand
conda install nvidia::cudnn

```


CMAKE:
`cuda_include_directories` to `include_directories`
`cuda_add_library` to `add_library` and maybe also `get_target_properties(<name> PROPERTIES LINKER_LANGUAGE CUDA)`
`include(FindCUDA)` to 
```
include(FindCUDAToolkit)
set(CUDA_FOUND TRUE)
```

`pip install nvtx`

replace find libs with:
```
      find_library(CUDA_CUDA_LIBRARY cuda
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        "C:/Users/Shadow/.mamba/envs/isaacsim/lib/x64")
      find_library(CUDA_CUDART_LIBRARY cudart
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        "C:/Users/Shadow/.mamba/envs/isaacsim/lib/x64")
      find_library(CUDA_NVRTC_LIBRARY nvrtc
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        "C:/Users/Shadow/.mamba/envs/isaacsim/lib/x64")
      find_library(CUDA_CUDNN_LIBRARY cudnn
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        "C:/Users/Shadow/.mamba/envs/isaacsim/Library/lib")
      find_library(CUDA_CUBLAS_LIBRARY cublas
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        "C:/Users/Shadow/.mamba/envs/isaacsim/lib/x64")
      find_library(CUDA_CURAND_LIBRARY curand
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        "C:/Users/Shadow/.mamba/envs/isaacsim/lib/x64")
```


```
cd dgl
md build
cd build

cmake -DCMAKE_CXX_FLAGS="/DDGL_EXPORTS" -DCMAKE_CONFIGURATION_TYPES="Release" -DDMLC_FORCE_SHARED_CRT=ON -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" -DCUDA_INCLUDE_DIRS="C:/Users/Shadow/.mamba/envs/isaacsim/include;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include" .. -G "Visual Studio 17 2022"

"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\msbuild" dgl.sln /m

CD ..\python

python setup.py install
```






