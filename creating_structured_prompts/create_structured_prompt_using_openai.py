import pandas as pd
import requests
import argparse
import os
import base64
from openai import OpenAI
from tqdm import tqdm
import time

openai_key = ""

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def create_image_for_caption(list_of_images):
    content = []
    for elem in list_of_images:
        content += [{"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode_image(elem)}"}},]
    return content


def prompt_gpt_for_structured_caption(caption, images_for_caption):
    client = OpenAI(api_key = openai_key)
    response = client.chat.completions.create(
        model = 'gpt-4o-mini',
        messages = [
            {
                'role': 'user',
                'content': [
                    {"type": "text",
                     "text": "I want to know what sound might be in the given scene as a caption and the associated images and you need to give me the results in the following format:"
                    },

                    {"type": "text",
                     "text": "Question: A baby cries while someone on TV screams",
                    },
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode_image('train/images/A_baby_cries_while_someone_on_TV_screams_image_1.jpg')}"}},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode_image('train/images/A_baby_cries_while_someone_on_TV_screams_image_5.jpg')}"}},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode_image('train/images/A_baby_cries_while_someone_on_TV_screams_image_10.jpg')}"}},
                    {"type": "text",
                     "text": "Answer: <someone on TV screams& start>@<baby cries& end>@<someone rides roller coaster& all>",
                    },
                    
                    {"type": "text",
                     "text": "Question: A cat meowing in terror while ladies talk and a bird squawks, ending with a short silence",
                    },
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode_image('train/images/A_cat_meowing_in_terror_while_ladies_talk_and_a_bird_squawks,_ending_with_a_short_silence_image_1.jpg')}"}},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode_image('train/images/A_cat_meowing_in_terror_while_ladies_talk_and_a_bird_squawks,_ending_with_a_short_silence_image_5.jpg')}"}},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode_image('train/images/A_cat_meowing_in_terror_while_ladies_talk_and_a_bird_squawks,_ending_with_a_short_silence_image_10.jpg')}"}},
                    {"type": "text",
                     "text": "Answer: <ladies talk& all>@<cat meows in terror& start>@<bird squaks& mid>@<silence& end>",
                    },
                    

                    {"type": "text",
                     "text": "Question: A male voice talking on television when a kitten meows followed by a woman and then a child talking",
                    },
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode_image('train/images/A_male_voice_talking_on_television_when_a_kitten_meows_followed_by_a_woman_and_then_a_child_talking_image_1.jpg')}"}},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode_image('train/images/A_male_voice_talking_on_television_when_a_kitten_meows_followed_by_a_woman_and_then_a_child_talking_image_5.jpg')}"}},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode_image('train/images/A_male_voice_talking_on_television_when_a_kitten_meows_followed_by_a_woman_and_then_a_child_talking_image_10.jpg')}"}},
                    {"type": "text",
                     "text": "Answer: <male voice talking on television& all>@<kitten meows& start>@<woman talks& mid>@<child talks& end>",
                    },

                    {"type":"text",
                     "text": f'All indicates the sound exists in the whole scene\
                              Start, mid, end indicates the time period the sound appear.\
                              Question: {caption}'}
                ]
                + images_for_caption +
                [
                    {"type": "text",
                    "text": "Answer: ",}
                ]
            }
        ],
        max_tokens = 300, 
        temperature = 0,
    )
    return response.choices[0].message.content



def create_dataframe(folder, dictionary_done):
    data = []
    audio_wav_files = [(os.path.splitext(file)[0], file) for file in os.listdir(folder + "audio_wav/") if file.endswith(('.wav'))]
    for item in tqdm(audio_wav_files):
        name, path = item
        caption = name.replace("_", " ")
        row = {
            "Caption": caption,
            "Structured_Caption": "",
            "Audio_mp3": "",
            "Audio_wav": path,
            "Image_1": "",
            "Image_2": "",
            "Image_3": "",
            "Video": "",
        }
        video_file = f"{name}.mp4"
        if video_file in os.listdir(folder + "videos/"):
            row['Video'] = video_file

        audio_mp3_file = f"{name}.mp3"
        if audio_mp3_file in os.listdir(folder + "audio_mp3/"):
            row['Audio_mp3'] = audio_mp3_file


        images = [img for img in os.listdir(folder + "images/") if img.startswith(name) and img.endswith(('.jpg'))]
        images.sort()
        caption_images = []
        if len(images) > 0:
            row['Image_1'] = images[0]
            row['Image_2'] = images[len(images)//2]
            row['Image_3'] = images[-1]
            caption_images = create_image_for_caption([folder + "images/" + images[0], folder + "images/" + images[len(images)//2], folder + "images/" + images[-1]])

        structured_caption = ""
        if caption in dictionary_done:
            structured_caption = dictionary_done[caption]
        else:
            structured_caption = prompt_gpt_for_structured_caption(caption, caption_images)
        
        row['Structured_Caption'] = structured_caption
        data.append(row)

        print(f"{caption} : {structured_caption}")

        #time.sleep(1)
    
    return data
    


if __name__ == '__main__':
    '''
    dictionary_of_done_values = {}
    with open("training_values.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        list_line = line.split(" : ")
        dictionary_of_done_values[list_line[0]] = list_line[1]

    folder = "train/"
    training_data = create_dataframe(folder, dictionary_of_done_values)
    train_df = pd.DataFrame(training_data)
    train_df.to_csv(folder + "train_df.csv", index=False)
    '''


    dictionary_of_done_values = {}
    with open("testing_values.txt", "r") as f:
        lines = f.readlines()
    for line in lines:
        list_line = line.split(" : ")
        dictionary_of_done_values[list_line[0]] = list_line[1]

    folder = "test/"
    testing_data = create_dataframe(folder, dictionary_of_done_values)
    test_df = pd.DataFrame(testing_data)
    test_df.to_csv(folder + "test_df.csv", index=False)


