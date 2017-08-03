from flask import Flask, render_template, request
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES, ALL



import zipfile
import os
import pickle
from keras import backend as K
import cv2
import h5py
import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import gensim
import sys
import matplotlib.pyplot as plt
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from scipy import spatial


word2vec_model = gensim.models.Word2Vec.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
with open('models/model_load.pkl', 'r') as f:
    model_load = pickle.load(f)

with open('models/model_train.pkl', 'r') as f:
    model_train = pickle.load(f)





## define look-up table
lookup = {0:'beach', 1:'bridge', 2:'cavern', 3:'cliff', 4:'dam', 5:'desert', 6:'farmland', 7:'glacier', 8:'island', 9:'lake', 10:'market', 11:'marsh', 12:'monument', 13:'mountain', 14:'ocean', 15:'prairie', 16:'rainforest', 17:'road', 18:'skyscraper', 19:'stadium', 20:'taiga', 21:'tundra', 22:'volcano', 23:'waterfall'}

##3
def avg_feature_vector(words, model, num_features):
        #function to average all words vectors in a given paragraph
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0

        #list containing names of words in the vocabulary
        #index2word_set = set(model.index2word) this is moved as input param for performance reasons
        for word in words:
            try:
                featureVec = np.add(featureVec, model[word])
                nwords = nwords+1
            except:
                pass

        if(nwords>0):
            featureVec = np.divide(featureVec, nwords)
        return featureVec


def return_locations(test_data_dir):

    # dimensions of our images.
    img_width, img_height = 224, 224

    # number of categories
    nb_category = 24

    # set theano order
    K.set_image_dim_ordering('th')

    # set image generator
    datagen = ImageDataGenerator(rescale=1./255)
    
    ##2 generate testing data(user image--intermediate stage)

    test_data_gen_folder = test_data_dir + 'test/' # there must be a subfolder(generator requirement)
    test_data_gen = os.listdir(test_data_gen_folder)
    test_data_gen = [test_d for test_d in test_data_gen if ('.jpg' in test_d)|('.JPG' in test_d)]
    nb_test_images = len(test_data_gen)
    
    generator = datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_width, img_height),
            batch_size=32,
            class_mode=None,
            shuffle=False)

    test_transformed = model_load.predict_generator(generator, nb_test_images)


    ##3 multiple images average predict using image-generator instead of direct loading

    mp_predict_proba = np.zeros([nb_test_images, nb_category])

    for i in range(nb_test_images):
        test_transformed_sg = test_transformed[i,:,:,:]
        test_transformed_sg = np.expand_dims(test_transformed_sg, axis=0)
        predict_proba = model_train.predict(test_transformed_sg)
        mp_predict_proba[i,:] = predict_proba

    avg_predict_proba = mp_predict_proba.mean(axis = 0)
    locs = map(lambda x:lookup[x], range(nb_category))

    proba_order = zip(avg_predict_proba.tolist(), locs)
    proba_order_sorted = sorted(proba_order, reverse=True)

    ##
    blend_loc = [proba[1] for proba in proba_order]
    blend_prob = [proba[0] for proba in proba_order]

    ##3 process input: blend_loc and blend_prob
    df = pd.read_csv('models/destinations_2.csv')
    df = df.drop('Unnamed: 0', axis=1)


    #---------- find closest vec in vecs to target_vec: top_ix
    num_features=300
    target_vec = np.zeros(num_features)
    #blend_loc = ['desert', 'ocean']
    #blend_prob = [0.5, 0.5]


    for loc, prob in zip(blend_loc, blend_prob):
        target_vec += prob * avg_feature_vector(loc.split(), model=word2vec_model, num_features=300)

    #---------- create vecs 1001x300 from locations
    vecs = np.zeros([df.shape[0], 300])
    vecs_sr = df['details'].apply(lambda x: avg_feature_vector(x.split(), model=word2vec_model, num_features=300))

    vect = []
    for vec in vecs_sr:
        vect.extend(vec)
    vecs = np.array(vect).reshape([df.shape[0], 300])


    #---------- calculate the distance between target_vec and 1001 location vecs
    target_to_vec_list = []
    for vec in vecs:
        target_to_vec = 1 - spatial.distance.cosine(target_vec ,vec)
        target_to_vec_list.append(target_to_vec)

    top_choices = sorted(zip(target_to_vec_list, range(len(target_to_vec_list))), key = lambda x:-x[0])
    top_ix  = [top_c[1] for top_c in top_choices]

    bottom_choices = sorted(zip(target_to_vec_list, range(len(target_to_vec_list))), key = lambda x:x[0])
    bottom_ix = [bottom_c[1] for bottom_c in bottom_choices]


    #----------- create name and pic links to top locations
    top_all_photo_list = []
    top_all_name_list = []
    bottom_all_photo_list = []
    bottom_all_name_list = []
    top_num_to_choose = 3
    bottom_num_to_choose = 3

    # calculate top places
    for rank_ix in range(top_num_to_choose):
        rank_name = df.loc[top_ix[rank_ix], 'location']
        rank_name = rank_name.replace('\x91_', ' ')

        rank_details = df.loc[top_ix[rank_ix], 'details']

        selection_list = []
        image_list = os.listdir('static/Images_selected/')
        for im in image_list:
            if rank_name in im.replace('\xc2\xa0\xc2\xa0','  '):
                selection_list.append(im)

        photo_total_num = len(selection_list)
        photo_list = [('static/Images_selected/' + selection) for selection in selection_list]

        if len(photo_list)>3:
            photo_list = photo_list[:3]

        top_all_photo_list.append(photo_list)
        top_all_name_list.append(rank_name)


    # calculate bottom(opposite) places
    for rank_ix in range(bottom_num_to_choose):
        rank_name = df.loc[bottom_ix[rank_ix], 'location']
        rank_name = rank_name.replace('\x91_', ' ')

        rank_details = df.loc[bottom_ix[rank_ix], 'details']

        selection_list = []
        image_list = os.listdir('static/Images_selected/')
        for im in image_list:
            if rank_name in im.replace('\xc2\xa0\xc2\xa0','  '):
                selection_list.append(im)

        photo_total_num = len(selection_list)
        photo_list = [('static/Images_selected/' + selection) for selection in selection_list]

        if len(photo_list)>3:
            photo_list = photo_list[:3]

        bottom_all_photo_list.append(photo_list)
        bottom_all_name_list.append(rank_name)

   
    return top_all_photo_list, top_all_name_list, bottom_all_photo_list, bottom_all_name_list, proba_order_sorted



app = Flask(__name__)

photos = UploadSet('photos', ALL)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.route('/', methods=['GET', 'POST'])
def index():

    top_all_photo_list = []
    top_all_name_list = []
    bottom_all_photo_list = []
    bottom_all_name_list = []
    proba_order_sorted = []
    str1 = ''
    str2 = ''

    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        str1 = 'YOU MAY ALSO LIKE...'
        str2 = 'NEW ADVENTURES...'
        # Unzip the files
        # Run the model
        try:
            with zipfile.ZipFile("static/img/" + filename) as zf:
                #print(zf.namelist())
                #print(zf.extractall("static/uploaded_files/" ))
                unzipped_dir = 'static/uploaded_files/' + zf.namelist()[0]
                test_data_dir = unzipped_dir + 'test/'

                if not os.path.exists(test_data_dir):
                    os.makedirs(test_data_dir)
                else:
                    shutil.rmtree(test_data_dir)
                    os.makedirs(test_data_dir)

                copy_origin = os.listdir(unzipped_dir)
                copy_origin = [copy for copy in copy_origin if ('.jpg' in copy) | ('.JPG' in copy)]

                print copy_origin

                for copy in copy_origin:
                    shutil.copy(unzipped_dir+copy, test_data_dir+copy)

                top_all_photo_list, top_all_name_list,bottom_all_photo_list, bottom_all_name_list,proba_order_sorted = return_locations(unzipped_dir)

        except zipfile.BadZipfile as e:
            print e

            upload_dir = 'static/img/'
            unzipped_dir = 'static/img/test/'
            test_data_dir = 'static/img/test/test/'

            if not os.path.exists(test_data_dir):
                os.makedirs(test_data_dir)
            else:
                shutil.rmtree(test_data_dir)
                os.makedirs(test_data_dir)

            shutil.copy(upload_dir+filename, test_data_dir+filename)
            
            top_all_photo_list, top_all_name_list,bottom_all_photo_list, bottom_all_name_list,proba_order_sorted = return_locations(unzipped_dir)

            #return "", 406
        # Run Model and generate Recommendations
    return render_template('index.html', similarities = [zip(top_all_photo_list, top_all_name_list,bottom_all_photo_list, bottom_all_name_list), proba_order_sorted, str1, str2])

if __name__ == '__main__':
    app.run(debug=False)
