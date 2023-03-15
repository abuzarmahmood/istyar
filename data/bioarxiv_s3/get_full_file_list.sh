for DIR in $(cat merged_dirs.txt);
do echo $DIR
    s3cmd ls $DIR --requester-pays >> full_file_list.txt
done
