from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='flickr8k',
                       karpathy_json_path='/data4/home/manikantab/Diffusion_models/NLP/a-PyTorch-Tutorial-to-Image-Captioning/caption_data/dataset_flickr8k.json',
                       image_folder='/data4/home/manikantab/Diffusion_models/NLP/a-PyTorch-Tutorial-to-Image-Captioning/flickr8k/images/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/data4/home/manikantab/Diffusion_models/NLP/a-PyTorch-Tutorial-to-Image-Captioning/output_folder/',
                       max_len=50)
