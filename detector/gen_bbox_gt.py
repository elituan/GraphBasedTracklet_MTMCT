import os
import sys
sys.path.append('../')
from config import cfg
from pathlib import Path
import pickle
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split



def get_gt_mot(gt, mots):
    # Prepare for extract MOT infor
    lines = [line.split(',') for line in gt.read_text().split("\n")][:-1]
    oid_unique = list(sorted(set([line[1] for line in lines])))
    line_chunks = []
    for oid in oid_unique:
        tmp = [line for line in lines if line[1] == oid]
        line_chunks.append(tmp)
    camId = gt.parent.parent.name
    sceId = gt.parent.parent.parent.name
    dirId = gt.parent.parent.parent.parent.name

    for _, line_chunk in enumerate(line_chunks):
        tracklet = {}
        frame_list = []
        for bboxID, line in enumerate(line_chunk):
            # extract information for mots dic
            frameId, oid, t, l, w, h, _, _, _, _ = line
            frame = "img%06d" % (int(frameId) - 1,)
            frame_list.append(int(frameId)-1)
            tid = "tid%03d" % (int(oid),)
            bboxid_ = "bbid%03d" % (bboxID,)
            tracklet[bboxID] = {'bbox': (int(t), int(l), int(w), int(h)),
                                'frame': frame,
                                'tid': oid,
                                'imgname': '{}_{}_{}_{}.jpg'.format(tid, camId, 1, bboxid_)
                                }

        dir_sid_cam_tid = (dirId, sceId, camId, oid)
        mots[dir_sid_cam_tid] = {
            'dirId': dirId,
            'sceId': sceId,
            'cam': camId,
            'tid': oid,
            'frame_list': frame_list,
            'tracklet': tracklet
        }
    return mots

if __name__ == '__main__':
    cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()

    # Get gt path
    data_path = "../datasets/AIC21_Track3_MTMC_Tracking/"
    chosen_dirs = cfg.SEC_DIR
    p = Path(data_path)
    gts = []
    for chosen_dir in chosen_dirs:
        gts.extend(list(sorted(p.glob(f'{chosen_dir}/*/*/gt/gt.txt'))))

    # get gt mot
    mots = dict()
    for gt in gts:
        mots = get_gt_mot(gt, mots)

    # train test split
    mot_key = list(mots.keys())
    # print(len(mot_key))
    mot_key_train, mot_key_test, _, _ = train_test_split(mot_key, mot_key, test_size=0.1, random_state=21)

    # Prepare for saving image
    src_path = Path('../datasets/detection/images')
    dst_train_path = Path('../reid/TransReID/data/aic21/image_train')
    if not dst_train_path.exists():
        dst_train_path.mkdir(parents=True, exist_ok=True)
    dst_test_path = Path('../reid/TransReID/data/aic21/image_test')
    if not dst_test_path.exists():
        dst_test_path.mkdir(parents=True, exist_ok=True)
    dst_query_path = Path('../reid/TransReID/data/aic21/image_query')
    if not dst_query_path.exists():
        dst_query_path.mkdir(parents=True, exist_ok=True)

    ## Save image of bbox
    train_paths = []
    test_paths = []
    for mot_key in tqdm(mot_key_test):
        tracklet = mots[mot_key]['tracklet']
        dirId = mots[mot_key]['dirId']
        sceId = mots[mot_key]['sceId']
        cam = mots[mot_key]['cam']

        # if cam != 'c001': continue

        test_bbox_paths = []
        for _, bbox in tracklet.items():
            frame, tlwh, imgname = bbox['frame'], bbox['bbox'], bbox['imgname']
            t, l, w, h = tlwh
            img_frame_path = src_path.joinpath(dirId, sceId, cam, 'img1', frame + '.jpg')
            img_frame = cv2.imread(str(img_frame_path))
            dst_img_path = dst_test_path.joinpath(imgname)
            #         print (img_frame_path)
            cv2.imwrite(str(dst_img_path), img_frame[l:l + h, t:t + w])
            test_bbox_paths.append([dst_img_path, tlwh])
        test_paths.append(test_bbox_paths)

    for mot_key in tqdm(mot_key_train):
        tracklet = mots[mot_key]['tracklet']
        dirId = mots[mot_key]['dirId']
        sceId = mots[mot_key]['sceId']
        cam = mots[mot_key]['cam']

        # if cam != 'c001': continue

        train_bbox_paths = []
        for _, bbox in tracklet.items():
            frame, tlwh, imgname = bbox['frame'], bbox['bbox'], bbox['imgname']
            t, l, w, h = tlwh
            img_frame_path = src_path.joinpath(dirId, sceId, cam, 'img1', frame + '.jpg')
            img_frame = cv2.imread(str(img_frame_path))
            dst_img_path = dst_train_path.joinpath(imgname)
            #         print (img_frame_path)
            cv2.imwrite(str(dst_img_path), img_frame[l:l + h, t:t + w])
            train_bbox_paths.append([dst_img_path, tlwh])
        train_paths.append(train_bbox_paths)

    # Save datatrain.pickle
    datatrain_pickle_path = Path('../reid/TransReID/data/aic21/datatrain.pickle')
    train_test_paths = {
        'train_paths': train_paths,
        'test_paths': test_paths
    }
    pickle.dump(train_test_paths, datatrain_pickle_path.open('wb'))