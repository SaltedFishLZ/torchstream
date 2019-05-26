mkdir -p ~/Datasets/Weizmann/Weizmann-avi
mkdir -p ~/Datasets/Weizmann/Weizmann-jpg

cd ~/Datasets/Weizmann/Weizmann-avi
wget http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/walk.zip
unzip walk.zip

wget http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/run.zip
unzip run.zip

wget http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/jump.zip
unzip run.zip

wget http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/side.zip
unzip side.zip

wget http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/bend.zip
unzip bend.zip

wget http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/wave1.zip
unzip wave1.zip

wget http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/wave2.zip
unzip wave2.zip

wget http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/pjump.zip
unzip pjump.zip

wget http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/jack.zip
unzip jack.zip

wget http://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/skip.zip
unzip skip.zip

rm *.zip