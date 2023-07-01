import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import skimage

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
data_folder = '/data4/home/manikantab/Diffusion_models/NLP/a-PyTorch-Tutorial-to-Image-Captioning/output_folder'  # folder with data files saved by create_input_files.py
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'  # base name shared by data files
dataset = "flickr8k"
output_folder = "output_folder"
captions_per_image = 5
min_word_freq = 5

def visualize_alphas(image_path, caption, alphas, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param caption: caption
    :param alphas: weights
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 16, 14 * 16], Image.LANCZOS)

    print(int(np.ceil((len(caption)/5.0))), len(caption))
    plt.subplot(int(np.ceil((len(caption))/5.0)), 5, 1)

    plt.text(0, 1, '%s' % (caption[0]), color='black', backgroundcolor='white', fontsize=8)
    plt.imshow(image)
    for t in range(1,len(caption)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(caption)/5.0)), 5, t + 1)
        plt.text(0, 1, '%s' % (caption[t]), color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(image)
        current_alpha = alphas[t-1, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.cpu().detach().numpy(), upscale=16, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.cpu().detach().numpy(), [14 * 16, 14 * 16])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)

        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.savefig('flickr8k/predictions/alphas_img_{}.jpg'.format(6))
    plt.show()

def test(img_path, img):
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    #index_map = {index: word for word, index in word_map.items()}
    # Create a base/root name for all output files
    #base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    #with open(os.path.join(output_folder, 'INDEXMAP_' + base_filename + '.json'), 'w') as j:
    #    json.dump(index_map, j)

    index_map_file = os.path.join(data_folder, 'INDEXMAP_' + data_name + '.json')
    with open(index_map_file, 'r') as j:
        index_map = json.load(j)

    checkpoint =  torch.load("flickr8k/checkpoints/BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar")
    decoder = checkpoint['decoder']
    encoder = checkpoint['encoder']
    #encoder = torch.load("flickr8k/trained_models/encoder")
    #decoder = torch.load("flickr8k/trained_models/decoder")

    decoder = decoder.to(device)
    encoder = encoder.to(device)
    img = img.to(device) # (3,224,224)

    # Encoder
    encoder_out = encoder(img.unsqueeze(0)) # (1,enc_img_size,enc_img_size,encoder_dim)


    # Decoder
    batch_size = encoder_out.size(0)
    encoder_dim = encoder_out.size(-1)
    enc_img_size = encoder_out.size(1)
    vocab_size = len(word_map)

    # Flatten image
    encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    caption = ["<start>"]
    last_word = word_map['<start>']

    # Embedding
    embeddings = decoder.embedding(torch.tensor([last_word]).to(device)) # (1,embed_dim)

    # Initialize LSTM state
    h, c = decoder.init_hidden_state(encoder_out)  # (1, decoder_dim)
    alphas = []
    while(last_word != word_map["<end>"]):
        attention_weighted_encoding, alpha = decoder.attention(encoder_out,h)
        # awe.shape = (1,encoder_dim)
        # alpha.shape = (1,num_pixels(14*14)) 
        alphas.append(alpha.view(enc_img_size,enc_img_size))
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (1, encoder_dim)
        attention_weighted_encoding = gate * attention_weighted_encoding

        h, c = decoder.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1), # (1, embed_dim+encoder_fim)
                (h, c))
        
        preds = decoder.fc(decoder.dropout(h))  # (1, vocab_size)
        pred = torch.argmax(preds).item()
        last_word = pred
        caption.append(index_map[str(pred)])
        embeddings = decoder.embedding(torch.tensor([last_word]).to(device)) # (1,embed_dim)


    alphas = torch.stack(alphas)
    print("alphas_shape : ",alphas.shape)
    visualize_alphas(img_path,caption, alphas)
    return ' '.join(caption)



def process(img_path):
    img = Image.open(img_path)
    #plt.imshow(np.array(img))
    #plt.show()
    #img.show()
    img = img.resize((256, 256))
    img = np.array(img)
    img = img.transpose(2, 0, 1)
    assert img.shape == (3, 256, 256)
    assert np.max(img) <= 255

    # Add random noise to the image array for probabilistic captions
    noise = np.random.normal(loc=0, scale=1, size=img.shape).astype(np.uint8)
    img = np.clip(img + noise, 0, 255)


    transform=transforms.Compose([transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    img = torch.FloatTensor(img/ 255.)
    img = transform(img)

    return img

if __name__ == '__main__':
    img_path = "flickr8k/images/3558796959_fc4450be56.jpg"
    img = process(img_path)
    print(test(img_path,img))