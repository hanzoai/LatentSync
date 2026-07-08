import argparse
import os
import tqdm
from statistics import fmean
from eval.syncnet import SyncNetEval
from eval.syncnet_detect import SyncNetDetector
from latentsync.utils.util import red_text
import torch


def syncnet_eval(syncnet, syncnet_detector, video_path, temp_dir, detect_results_dir="detect_results"):
    syncnet_detector(video_path=video_path, min_track=50)
    crop_videos = os.listdir(os.path.join(detect_results_dir, "crop"))
    if crop_videos == []:
        raise Exception(red_text(f"Face not detected in {video_path}"))
    av_offset_list = []
    conf_list = []
    dist_list = []
    for video in crop_videos:
        av_offset, min_dist, conf = syncnet.evaluate(
            video_path=os.path.join(detect_results_dir, "crop", video), temp_dir=temp_dir
        )
        av_offset_list.append(av_offset)
        conf_list.append(conf)
        dist_list.append(min_dist)
    av_offset = int(fmean(av_offset_list))
    conf = fmean(conf_list)
    min_dist = fmean(dist_list)
    print(f"Input video: {video_path}")
    print(f"LSE-C (SyncNet confidence): {conf:.4f}")
    print(f"LSE-D (SyncNet min distance): {min_dist:.4f}")
    print(f"AV offset: {av_offset}")
    return av_offset, min_dist, conf


def main():
    parser = argparse.ArgumentParser(description="SyncNet")
    parser.add_argument("--initial_model", type=str, default="checkpoints/auxiliary/syncnet_v2.model", help="")
    parser.add_argument("--video_path", type=str, default=None, help="")
    parser.add_argument("--videos_dir", type=str, default="/root/processed")
    parser.add_argument("--temp_dir", type=str, default="temp", help="")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    syncnet = SyncNetEval(device=device)
    syncnet.loadParameters(args.initial_model)

    syncnet_detector = SyncNetDetector(device=device, detect_results_dir="detect_results")

    if args.video_path is not None:
        syncnet_eval(syncnet, syncnet_detector, args.video_path, args.temp_dir)
    else:
        sync_conf_list = []
        video_names = sorted([f for f in os.listdir(args.videos_dir) if f.endswith(".mp4")])
        for video_name in tqdm.tqdm(video_names):
            try:
                _, _, conf = syncnet_eval(
                    syncnet, syncnet_detector, os.path.join(args.videos_dir, video_name), args.temp_dir
                )
                sync_conf_list.append(conf)
            except Exception as e:
                print(e)
        print(f"The average sync confidence is {fmean(sync_conf_list):.02f}")


if __name__ == "__main__":
    main()
