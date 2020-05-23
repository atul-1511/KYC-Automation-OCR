import numpy as np
import cv2
import argparse
import os
import pandas as pd 
import time

parser = argparse.ArgumentParser()
parser.add_argument('--nsamples',default=1000, help = 'No. of samples to be generated',type = int)
parser.add_argument('--write_folder',default = 'samples_generated', help = 'Destination folder to write generated samples', type = str)
parser.add_argument('--csv_name', default = 'sample_text.csv',help = 'Name of csv file', type = str)
parser.add_argument('--characters', default = 'abcdefghijklmnopqrst', help = 'Characters to be considered for text image generation',\
                    type = str)
parser.add_argument('--length',default = 10, help = 'No. of characters of text string in image', type = int)

arguments = parser.parse_args()

arguments.manualSeed = np.random.randint(1, 10000)
print("Random Seed: ", arguments.manualSeed)
np.random.seed(arguments.manualSeed)
def char_dictionary():
    char_dict = {}
    for ind, char in enumerate(characters) :
        char_dict[ind] = char       
    return char_dict
def random_text(length, random_length = False):
    if random_length:
        length = np.random.randint(5,10)
    else :
        length = length
    
    char_select = np.random.randint(0, characters_length-1, length) 
    text = ''.join(char_dict[i] for i in char_select)

    return text
def text_size(font,text, size) :
    return cv2.getTextSize(text, font, size, cv2.INTER_AREA)
def generate_image(font, get_random_font = False) :
    if get_random_font :
        font = fonts[np.random.randint(0,fonts_length-1)]
    text = random_text(length) 
    #print (text)
    text_size_generated, baseline = text_size(font = font, text = text, size = 2)
    img_height = text_size_generated[1] + baseline + 20
    img_width = text_size_generated[0]  + baseline + 20
    #INPUT THE BACKGROUND IMAGE HERE
    img = cv2.imread(r"C:\Users\u293217\Downloads\3.png",3)
    img = cv2.resize(img,(img_width,img_height))
    
    img = cv2.putText(img,text,(1,img_height - 10 - baseline), font, 2 , (0,0,0),cv2.INTER_AREA)
    return text, img 


if __name__ == '__main__' :
    print (__name__)
    characters = list(arguments.characters)
    characters_length = len(characters)
    fonts = [cv2.FONT_HERSHEY_COMPLEX,cv2.FONT_HERSHEY_COMPLEX_SMALL, \
        cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN, \
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, \
        cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC]
    fonts_length = len(fonts)
    samples_to_be_generated = arguments.nsamples 
    char_dict = char_dictionary()
    length = arguments.length
    write_folder = arguments.write_folder

    start = time.time()
    img_name = []
    img_text = []
    for i in range(arguments.nsamples) :
        text, img = generate_image(font = cv2.FONT_HERSHEY_DUPLEX)
        if not os.path.exists(write_folder) :
            os.mkdir(write_folder)
        cv2.imwrite(write_folder+'/img_{}.jpg'.format(i), img)
        img_name.append('img_{}.jpg'.format(i))
        img_text.append(text)

    print ('Total time taken: '+str(time.time()-start))
    data = pd.DataFrame({'Image_Name':img_name,'Text':img_text})
    data.to_csv(arguments.csv_name, index = False) 
