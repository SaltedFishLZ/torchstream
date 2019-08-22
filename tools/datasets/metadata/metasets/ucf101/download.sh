ZIP_DIR="${HOME}/Datasets/UCF101/UCF101-zip"
AVI_DIR="${HOME}/Datasets/UCF101/UCF101-avi"
JPG_DIR="${HOME}/Datasets/UCF101/UCF101-jpg"

mkdir -p ${ZIP_DIR}

cd ${ZIP_DIR}
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x UCF101.rar
mv UCF-101 UCF101-avi

if [ ! -d "$AVI_DIR" ]; then
    mv UCF101-avi "$AVI_DIR"
fi