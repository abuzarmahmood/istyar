#DIR=$(head -n 1 processed_full_file_list.txt)
DIR=$1

DATA_DIR=/media/bigdata/projects/istyar/data/bioarxiv_s3/raw
FILENAME=$(basename $DIR)
BASENAME=${FILENAME:0:36}
SAVE_DIR=$DATA_DIR/$BASENAME

if "$(cat downloaded_data.txt | wc -l)" -le 100
then
    if grep -Fxq $BASENAME downloaded_data.txt 
    then
        echo $BASENAME already processed
    else
        echo Processing : $BASENAME
        s3cmd get $DIR $DATA_DIR/$FILENAME --requester-pays
        unzip $DATA_DIR/$FILENAME -d $SAVE_DIR 
        rm $(find $SAVE_DIR -type f -not -iname "*.xml")
        rm $DATA_DIR/$FILENAME
        echo $BASENAME >> downloaded_data.txt
    fi
else
    echo Past Download Limit
fi
