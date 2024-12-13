import numpy as np
import pandas as pd


def create_tsv_train(filename):
	# Load the data
	data = pd.read_csv(filename)

	# Columns: Name, Dataset, Caption, Audio Path, Duration, mel_path

	# Create a new dataframe
	df = pd.DataFrame(columns=['name', 'dataset', 'ori_cap', 'audio_path', 'duration', 'mel_path'])

	# Iterate over the data
	for i in range(len(data)):
		# Get the duration
		duration = 10.0
		# Get the caption
		caption = data['Caption'][i].replace('""', '')
		# Get the name
		name = data['Audio_wav'][i]
		audio_path = f'data/audio_wav_train/{name}'
		name_mel = name.replace('.wav', '_mel')
		mel_path = f'data/audio_wav_train_mel/{name_mel}.npy'
		# Get the dataset
		dataset = "audiocaps"
		
		# Append to the dataframe
		df = df._append({'name': name, 'dataset': dataset, 'ori_cap': caption, 'audio_path': audio_path, 'duration': duration, 'mel_path': mel_path}, ignore_index=True)

	# Save the dataframe
	df.to_csv('train.tsv', sep='\t', index=False)

	#Add column named struct_cap to the dataframe, fill with data[Structured Caption]
	df['caption'] = data['Structured_Caption']

	# Save the dataframe
	df.to_csv('train_struct.tsv', sep='\t', index=False)


def create_tsv_test(filename):
	#columns: name, dataset, ori_cap, audio_path, mel_path, caption
	# Load the data
	data = pd.read_csv(filename)

	df = pd.DataFrame(columns=['name', 'dataset', 'ori_cap', 'audio_path', 'mel_path', 'caption'])

	for i in range(len(data)):
		# Get the caption
		caption = data['Caption'][i].replace('""', '')
		# Get the name
		name = data['Audio_wav'][i]
		audio_path = f'data/audio_wav_test/{name}'
		name_mel = name.replace('.wav', '_mel')
		mel_path = f'data/audio_wav_test_mel/{name_mel}.npy'
		# Get the dataset
		dataset = "audiocaps"
		struct_cap = data['Structured_Caption'][i]
		
		# Append to the dataframe
		df = df._append({'name': name, 'dataset': dataset, 'ori_cap': caption, 'audio_path': audio_path, 'mel_path': mel_path, 'caption': struct_cap}, ignore_index=True)

	# Save the dataframe
	df.to_csv('test.tsv', sep='\t', index=False)	


if __name__ == '__main__':
	create_tsv_train('../../creating_structured_prompts/train_df.csv')
	# create_tsv_test('../../creating_structured_prompts/test_df.csv')
