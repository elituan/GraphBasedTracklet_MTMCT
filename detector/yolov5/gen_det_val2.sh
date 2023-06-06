#seqs=(c006 c007 c008 c009 c010 c016 c017 c018 c019 c020 c021 c022 c023 c024 c025 c026 c027 c028 c029 c033 c034 c035 c036)
#seqs=(c006 c007 c008 c009 c010 c016 c017 c018 c019)
#seqs=(c020 c021 c022 c023 c024 c025 c026 c027 c028)
seqs=(c029 c033 c034 c035 c036)

gpu_id=0
for seq in ${seqs[@]}
do
    CUDA_VISIBLE_DEVICES=${gpu_id} python detect2img.py --name ${seq} --weights yolov5x.pt --conf 0.1 --agnostic --save-txt --save-conf --img-size 1280 --classes 2 5 7 --cfg_file $1&
    gpu_id=$(($gpu_id))
done
wait
