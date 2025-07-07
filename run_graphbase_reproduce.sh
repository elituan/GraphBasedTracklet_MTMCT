MCMT_CONFIG_FILE="graphbase_mcmt.yml"
#### Run Detector.####
cd detector/
python gen_images_aic.py ${MCMT_CONFIG_FILE}

cd yolov5/
sh gen_det.sh ${MCMT_CONFIG_FILE}


#### Extract reid feature.####
cd ../../reid/
python extract_image_feat.py "aic_reid_siamese.yml" #Siamese Network

#### MOT. ####
cd ../tracker/MOTBaseline
bash run_aic.sh ${MCMT_CONFIG_FILE}

#wait
#### Get results. ####
cd ../../reid/reid-matching/tools
python trajectory_fusion.py ${MCMT_CONFIG_FILE}
#note: sub cluster based on the simGNN result
python sub_cluster.py ${MCMT_CONFIG_FILE}
python gen_res.py ${MCMT_CONFIG_FILE}

#### Vis. (optional) ####
#python viz_mot.py ${MCMT_CONFIG_FILE}
python viz_mcmt.py ${MCMT_CONFIG_FILE}
