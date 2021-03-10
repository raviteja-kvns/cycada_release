# Download devkit
git clone https://github.com/mcordts/cityscapesScripts lib/dataset_devkit/cityscapesScripts

# Download coco format anntations

mkdir -p data/cityscapes/annotations

if [ ! -f data/cityscapes/annotations/instancesonly_gtFine_train.json ]; then
    curl http://www.yuwenxiong.com/dataset/cityscapes/annotations/instancesonly_gtFine_train.json -o data/cityscapes/annotations/instancesonly_gtFine_train.json
fi

if [ ! -f data/cityscapes/annotations/instancesonly_gtFine_val.json ]; then
    curl http://www.yuwenxiong.com/dataset/cityscapes/annotations/instancesonly_gtFine_val.json -o data/cityscapes/annotations/instancesonly_gtFine_val.json
fi

if [ ! -f data/cityscapes/annotations/cityscapes_fine_val.json ]; then
    curl http://www.yuwenxiong.com/dataset/cityscapes/annotations/cityscapes_fine_val.json -o data/cityscapes/annotations/cityscapes_fine_val.json
fi

cd data/cityscapes

if [ ! -d images ]; then
    mkdir images
    # Replacing the below command with creating softlinks
    # cp leftImg8bit/*/*/*.png images
    
    # Linking instead of copying
    # train images
    ln -s leftImg8bit/train/aachen/* images/
    ln -s leftImg8bit/train/bochum/* images/
    ln -s leftImg8bit/train/bremen/* images/
    ln -s leftImg8bit/train/cologne/* images/
    ln -s leftImg8bit/train/darmstaadt/* images/
    ln -s leftImg8bit/train/dusseldorf/* images/
    ln -s leftImg8bit/train/erfurt/* images/
    ln -s leftImg8bit/train/hamburg/* images/
    ln -s leftImg8bit/train/hanover/* images/
    ln -s leftImg8bit/train/jena/* images/
    ln -s leftImg8bit/train/krefeld/* images/
    ln -s leftImg8bit/train/monchengladbach/* images/
    ln -s leftImg8bit/train/strasbourg/* images/
    ln -s leftImg8bit/train/stuttgart/* images/
    ln -s leftImg8bit/train/tubingen/* images/
    ln -s leftImg8bit/train/ulm/* images/
    ln -s leftImg8bit/train/weimar/* images/
    ln -s leftImg8bit/train/zurich/* images/

    # Test Images
    ln -s leftImg8bit/test/berlin/* images/
    ln -s leftImg8bit/test/bielefeld/* images/
    ln -s leftImg8bit/test/bonn/* images/
    ln -s leftImg8bit/test/leverkusen/* images/
    ln -s leftImg8bit/test/mainz/* images/
    ln -s leftImg8bit/test/munich/* images/

    # Train extra Images
    ln -s leftImg8bit/train_extra/augsburg/* images/
    ln -s leftImg8bit/train_extra/bad-honnef/* images/
    ln -s leftImg8bit/train_extra/bamberg/* images/
    ln -s leftImg8bit/train_extra/bayreuth/* images/
    ln -s leftImg8bit/train_extra/dortmund/* images/
    ln -s leftImg8bit/train_extra/dresden/* images/
    ln -s leftImg8bit/train_extra/duisburg/* images/
    ln -s leftImg8bit/train_extra/erlangen/* images/
    ln -s leftImg8bit/train_extra/freiburg/* images/
    ln -s leftImg8bit/train_extra/heidlbreg/* images/
    ln -s leftImg8bit/train_extra/heilbronn/* images/
    ln -s leftImg8bit/train_extra/karlsruhe/* images/
    ln -s leftImg8bit/train_extra/konigswinter/* images/
    ln -s leftImg8bit/train_extra/konstanz/* images/
    ln -s leftImg8bit/train_extra/mannhim/* images/
    ln -s leftImg8bit/train_extra/muhlheim-ruhr/* images/
    ln -s leftImg8bit/train_extra/nuremberg/* images/
    ln -s leftImg8bit/train_extra/oberhausen/* images/
    ln -s leftImg8bit/train_extra/saarbrucken/* images/
    ln -s leftImg8bit/train_extra/schweinfurt/* images/
    ln -s leftImg8bit/train_extra/troisdorf/* images/
    ln -s leftImg8bit/train_extra/wuppertal/* images/
    ln -s leftImg8bit/train_extra/wurzburg/* images/

    # Val Images
    ln -s leftImg8bit/val/frankfurt/* images/
    ln -s leftImg8bit/val/lindau/* images/
    ln -s leftImg8bit/val/munster/* images/    

fi


if [ ! -d labels ]; then
    mkdir labels
    # cp gtFine/*/*/*labelTrainIds.png labels

    # Linking instead of copying
    ln -s gtFine/train/aachen/*labelTrainIds.png labels
    ln -s gtFine/train/bochum/*labelTrainIds.png labels
    ln -s gtFine/train/bremen/*labelTrainIds.png labels
    ln -s gtFine/train/cologne/*labelTrainIds.png labels
    ln -s gtFine/train/darmstaadt/*labelTrainIds.png labels
    ln -s gtFine/train/erfurt/*labelTrainIds.png labels
    ln -s gtFine/train/hamburg/*labelTrainIds.png labels
    ln -s gtFine/train/hanover/*labelTrainIds.png labels
    ln -s gtFine/train/jena/*labelTrainIds.png labels
    ln -s gtFine/train/krefeld/*labelTrainIds.png labels
    ln -s gtFine/train/monchengladbach/*labelTrainIds.png labels
    ln -s gtFine/train/strasbourg/*labelTrainIds.png labels
    ln -s gtFine/train/stuttgart/*labelTrainIds.png labels
    ln -s gtFine/train/tubingen/*labelTrainIds.png labels
    ln -s gtFine/train/ulm/*labelTrainIds.png labels
    ln -s gtFine/train/weimar/*labelTrainIds.png labels
    ln -s gtFine/train/zurich/*labelTrainIds.png labels

    # Test labels
    ln -s gtFine/test/berlin/*labelTrainIds.png labels
    ln -s gtFine/test/bielefeld/*labelTrainIds.png labels
    ln -s gtFine/test/bonn/*labelTrainIds.png labels
    ln -s gtFine/test/leverkusen/*labelTrainIds.png labels
    ln -s gtFine/test/mainz/*labelTrainIds.png labels
    ln -s gtFine/test/munich/*labelTrainIds.png labels

    # Val labels
    ln -s gtFine/val/frankfurt/*labelTrainIds.png labels
    ln -s gtFine/val/lindau/*labelTrainIds.png labels
    ln -s gtFine/val/munster/*labelTrainIds.png labels    

fi

if [ ! -d panoptic ]; then
    curl http://www.yuwenxiong.com/dataset/cityscapes/annotations/panoptic.zip -o panoptic.zip
    unzip -q panoptic.zip
fi

