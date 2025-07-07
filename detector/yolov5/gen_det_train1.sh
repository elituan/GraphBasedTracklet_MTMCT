#seqs=(c041 c042 c043 c044 c045 c046)
#seqs=(c001 c002 c003 c004 c005 c010 c011 c012 c013 c014 c015 c016 c017 c018 c019 c020 c021 c022 c023 c024 c025 c026 c027 c028 c029 c030 c031 c032 c033 c034 c035 c036 c037 c038 c039 c040)
#seqs=(c001 c002 c003 c004 c005 c010 c011 c012 c013)
seqs=(c014 c015 c016 c017 c018 c019 c020 c021 c022)
#seqs=(c023 c024 c025 c026 c027 c028 c029 c030 c031)
#seqs=(c032 c033 c034 c035 c036 c037 c038 c039 c040)

gpu_id=0
for seq in ${seqs[@]}
do
    CUDA_VISIBLE_DEVICES=${gpu_id} python detect2img.py --name ${seq} --weights yolov5x.pt --conf 0.1 --agnostic --save-txt --save-conf --img-size 1280 --classes 2 5 7 --cfg_file $1&
    gpu_id=$(($gpu_id))
done
wait
