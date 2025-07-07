############################## Train TransMtmc model with ground true label from CityflowV2 #########################

MCMT_CONFIG_FILE="aic_all_train.yml"
TransMtmc_CONFIG_FILE="configs/aic21/vit_transMtmc_stride.yml"

#### Run Detector.####
cd detector/
## save frames of all videos in [test, train, validation] into
## {datasets/detection/images/test/S06/c041/img1/img001111.jpg}
python gen_images_aic.py ${MCMT_CONFIG_FILE}

### Save ground true bbox (from gt file -> extract gt bbox -> save)
python gen_bbox_gt.py ${MCMT_CONFIG_FILE}

### Train TransMtmc
cd ../reid/TransReID
python train.py --config_file ${TransMtmc_CONFIG_FILE}
#CUDA_LAUNCH_BLOCKING=1 python train.py --config_file ${TransMtmc_CONFIG_FILE} MODEL.DEVICE_ID "('0')"


cd yolov5/
## detect bbox from saved frame. Save as pkl file in AIC21-MTMC/datasets/detect_merge/cam/{cam}_dets.pkl

# Train dir
#sh gen_det_train0.sh ${MCMT_CONFIG_FILE}
#sh gen_det_train1.sh ${MCMT_CONFIG_FILE}
#sh gen_det_train2.sh ${MCMT_CONFIG_FILE}
#sh gen_det_train3.sh ${MCMT_CONFIG_FILE}

# Val dir
#sh gen_det_train0.sh ${MCMT_CONFIG_FILE}
#sh gen_det_train1.sh ${MCMT_CONFIG_FILE}
#sh gen_det_train2.sh ${MCMT_CONFIG_FILE}

#### Extract reid feature.####
## Load the model, load the saved para, process bbox and
## save feat as pkl file in AIC21-MTMC/datasets/detect_merge/cam/{cam}_dets_feat.pkl
# todo: Add transformer module
#cd ../../reid/
#python extract_image_feat.py "aic_reid1.yml"
#python extract_image_feat.py "aic_reid2.yml"
#python extract_image_feat.py "aic_reid3.yml"
#python merge_reid_feat.py ${MCMT_CONFIG_FILE}

### MOT. ####
#cd ../tracker/MOTBaseline
# Extract MOT tracklets for each camera.
# Output is saved in {AIC21-MTMC/datasets/detect_merge/cam/{cam}_mot.txt} And {..._mot_feat.pkl}
#sh run_aic.sh ${MCMT_CONFIG_FILE}
#wait

### Get results. ####
# todo Add transformer for tracklets
cd ../../reid/reid-matching/tools

## Filter MOT, filter bbox
## Output: {AIC21-MTMC/reid/reid-matching/tools/exp/viz/test/S06/trajectory/c041.pkl}
## Output: {AIC21-MTMC/datasets/detect_merge/cam/cam_mot_feat_break.pkl}
#python trajectory_fusion.py ${MCMT_CONFIG_FILE}

## Output: {reid/reid-matching/tools/test_cluster.pkl}
#python sub_cluster.py ${MCMT_CONFIG_FILE}

#python gen_res.py ${MCMT_CONFIG_FILE}

#### Vis. (optional) ####
#python viz_mot.py ${MCMT_CONFIG_FILE}
#python viz_mcmt.py ${MCMT_CONFIG_FILE}
