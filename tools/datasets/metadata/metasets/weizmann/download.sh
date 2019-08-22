ZIP_DIR="${HOME}/Datasets/Weizmann/Weizmann-zip"
AVI_DIR="${HOME}/Datasets/Weizmann/Weizmann-avi"
JPG_DIR="${HOME}/Datasets/Weizmann/Weizmann-jpg"

labels=(
    walk
    run
    jump
    pjump
    side
    bend
    wave1
    wave2
    jack
    skip
)

cd ${ZIP_DIR}
for label in "${labels[@]}"
do
    wget "http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/${label}.zip"
    target_directory="${AVI_DIR}/${label}"
    mkdir -p ${target_directory}
    unzip "${label}.zip" -d ${target_directory}
done