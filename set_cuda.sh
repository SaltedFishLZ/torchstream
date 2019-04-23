#/bin/bash

VENV_PATH="venv_video"

# set CUDA 9.0 path
# if there is any error, please check your CUDA 9.0 installation path
echo "" >> ${VENV_PATH}/bin/activate
echo "# Set CUDA Path" >> ${VENV_PATH}/bin/activate
echo "export PATH=\"${PATH}:/usr/local/cuda-9.0/bin\"" >> \
${VENV_PATH}/bin/activate
echo "export LD_LIBRARY_PATH=\"${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64\""\
 >> ${VENV_PATH}/bin/activate

