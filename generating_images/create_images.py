import pandas as pd
import subprocess
#import youtube_dl
import os
from pydub import AudioSegment
from pytube import YouTube
#from moviepy.editor import AudioFileClip
import cv2
from tqdm import tqdm


def download_audio_and_images(folder_path, video_path, caption, start_time):
    
    #Save audio file
    audio_path = folder_path + "audio/" + caption + ".mp3"
    command = [
        "ffmpeg",
        "-i", video_path,
        "-ss", str(start_time),
        "-t", str(10),
        "-q:a", "0",
        "-map", "a",
        audio_path,
        "-y"
    ]
    subprocess.run(command, check=True)

    #Save 3 images
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    timestamps = [start_time + 0.5, start_time + 5, start_time + 10]
    for i, t in enumerate(timestamps):
        frame_number = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            image_path = os.path.join(f"{folder_path}images/{caption}_image_{i+1}.jpg")
            cv2.imwrite(image_path, frame)
            #print(f"Image saved to {image_path}")
        else:
            print(f"Failed to extract frame at {t} seconds")
    
    cap.release()


    return


def save_audio_and_image(dataframe, num_samples, file_path, all_video_files):
    processed_samples = 0
    for idx, row in tqdm(dataframe.iterrows()):
        id = row['youtube_id']
        start_time = row['start_time']
        caption = row['caption'].replace(" ", "_")
        
        youtube_url = "https://www.youtube.com/watch?v="+id
        #Save video file
        video_path = file_path + "videos/" + caption + ".mp4"
        if caption + ".mp4" not in all_video_files:
            result = subprocess.run(["yt-dlp", "-o", video_path, "--recode-video", "mp4", youtube_url])
            if result.returncode == 0:
                    download_audio_and_images(file_path, video_path, caption, start_time)
                    print(f"Saved audio, video and image files for file {idx}")
                    processed_samples += 1
        
        if processed_samples == num_samples:
            break

    return

def fix_image_generation_from_videos(path, dataframe):
    for video in tqdm(os.listdir(path + "videos/")):
        if ".mp4" in video:
            video_split = video.split(".")
            if len(video_split) > 2:
                caption = video_split[0].replace("_", " ") + "." + video_split[1].replace("_", " ")
            else:
                caption = video_split[0].replace("_", " ")
            row = dataframe.loc[dataframe['caption'] == caption]
            start_time = row.iloc[0]['start_time']
            cap = cv2.VideoCapture(path + "videos/" + video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            timestamps = [start_time + i for i in range(10)]
            for i, t in enumerate(timestamps):
                frame_number = int(t * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if ret:
                    caption = caption.replace(" ", "_")
                    image_path = os.path.join(f"{path}images/{caption}_image_{i+1}.jpg")
                    print(f"Saving to {image_path}")
                    cv2.imwrite(image_path, frame)
                    #print(f"Image saved to {image_path}")
                else:
                    print(video)
                    print(f"Failed to extract frame at {t} seconds")
    
    cap.release()
            
    return

def convert_mp3_to_wav(input_folder, output_folder):
    for file in tqdm(os.listdir(input_folder)):
        file_name = file.split(".")[0]
        wav_file = file_name + ".wav"
        try:
            sound = AudioSegment.from_mp3(input_folder + file)
            sound.export(output_folder + wav_file, format="wav")
        except:
            print(file)
            pass
    return





if __name__ == "__main__":
    
    all_video_in_train = [f for f in os.listdir("train/videos/") if os.path.isfile(os.path.join("train/videos/", f))]
    all_video_in_test = [f for f in os.listdir("test/videos/") if os.path.isfile(os.path.join("test/videos/", f))]
    all_video_files = all_video_in_train + all_video_in_test


    num_train = 2001 - len(all_video_in_train) 
    train_df = pd.read_csv("train.csv")
    #save_audio_and_image(train_df, num_train, "train/", all_video_files)
    fix_image_generation_from_videos("train/", train_df)
    #convert_mp3_to_wav("train/audio/", "train/audio_wav/")
   
    
    num_test = 201 - len(all_video_in_test)
    test_df = pd.read_csv("test.csv")
    #save_audio_and_image(test_df, num_test, "test/", all_video_files)
    fix_image_generation_from_videos("test/", test_df)
    #convert_mp3_to_wav("test/audio/", "test/audio_wav/")
    

    


