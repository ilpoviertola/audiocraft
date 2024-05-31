import pytest
import torch
import json

from audiocraft.models.v2agen import V2AGen
from audiocraft.data.video_dataset import VideoDataset
from audiocraft.data.audio import audio_write


class TestV2AGenModel:
    def get_v2agen(self):
        v2agen = V2AGen.get_pretrained(
            filepath="/home/hdd/ilpo/logs/audiocraft/xps/23e17883/checkpoint_50.th",
            device="cuda",
        )
        return v2agen

    def test_base(self):
        v2agen = self.get_v2agen()
        assert v2agen.vfps == 25
        assert v2agen.sample_rate == 24000
        assert v2agen.audio_channels == 1

    def test_generate_with_video_condition(self):
        v2agen = self.get_v2agen()
        v2agen.set_generation_params(duration=4.48)
        video_dir = "/home/hdd/ilpo/datasets/greatesthit/vis-data-256_h264_video_25fps_256side_24000hz_aac_len_5_splitby_random/"
        filenames = [
            "2015-10-02-11-46-40-222_denoised_795.mp4",
            "2015-09-27-22-43-25-1_denoised_1041.mp4",
            "2015-03-28-19-43-23_denoised_248.mp4",
            "2015-09-23-15-20-31-666_denoised_156.mp4",
            "2015-09-29-13-27-53-50_denoised_191.mp4",
            "2015-09-29-15-44-54-400_denoised_634.mp4",
            "2015-10-06-21-13-36-13_denoised_509.mp4",
            "2015-03-29-02-06-05_denoised_461.mp4",
            "2015-09-27-22-43-25-227_denoised_835.mp4",
            "2015-03-31-01-55-05_denoised_241.mp4",
        ]
        videos = []
        for file in filenames:
            video, _, _ = VideoDataset._read_video(
                fn=f"{video_dir}/{file}",
                start_pts=0.0,
                end_pts=4.48,
            )
            videos.append(video)
        wavs = v2agen.generate_with_video(videos)
        for i, wav in enumerate(wavs):
            audio_write(f"./test_files/{filenames[i][:-4]}", wav[0], 24000)
            meta = {
                "conditioning": {
                    "video": {
                        "path": [f"{video_dir}/{filenames[i]}"],
                        "seek_time": [0.0],
                    }
                }
            }
            json.dump(meta, open(f"./test_files/{filenames[i][:-4]}.json", "w"))
