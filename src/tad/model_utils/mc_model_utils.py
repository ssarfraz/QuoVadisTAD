import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from tad.evaluation_scripts.evaluate import evaluate
from tad.utils.data_utils import median_iqr_norm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MCDropOutLayer(layers.Layer):
    def __init__(self, mc_rate, mc_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_rate = mc_rate
        self.mc_steps = mc_steps
    
    def call(self, inputs):
        input_, train = inputs
        dropouts_ = [layers.Dropout(self.mc_rate)(input_, training=train) for _ in range(self.mc_steps)]        
        return dropouts_
    

def mc_drop_out_layer(embeddings, mc_rate=0.5, mc_steps=10):
    #batch_size = tf.shape(embeddings)[0]
    #feat_dim = tf.shape(embeddings)[1]
    dropouts_ = [layers.Dropout(mc_rate)(embeddings, training=True) for i in range(mc_steps)]
    dropouts_ = [embeddings] + dropouts_
    #dropouts_.insert(0, embeddings)  # to have the orignal predictions without dropout
    return dropouts_  #

def conv_model_blk(embeddings, input_shape):
    regression_targets = layers.GlobalAveragePooling2D(keepdims=True)(embeddings)
    return layers.Reshape((1, input_shape[1]))(regression_targets)
    

def load_model_mc(pretrained_model, input_shape=None, config=None):
    _, embeddings = pretrained_model.output
    dropouts_ = MCDropOutLayer(config['mc_dropout_rate'], config['mc_steps'])([embeddings, True])
    dropouts_ = [embeddings] + dropouts_
    #dropouts_ = layers.Lambda(mc_drop_out_layer)(embeddings)
    if config['model'] == "conv_model":        
        dropouts_ = [conv_model_blk(r, input_shape) for r in dropouts_]
    
    mc_regression_preds = tf.stack(dropouts_, axis=1)    # shape [batch_size, mc_stpes, 1, sensors]
    mc_regression_preds = layers.Dense(input_shape[1])(mc_regression_preds)
    
    return keras.Model(inputs=pretrained_model.input, outputs=[mc_regression_preds, embeddings])





def compute_entropy(probs):
        return -np.sum(probs * np.log(probs + 1e-6), axis=-1)


def uncertainty_estimation(mc_preds, targets):
    mc_dim = 1
    
    orig_preds = mc_preds[:, 0,  :]
    orig_error = np.abs(np.subtract(orig_preds, targets))
    
    # uncertainty estimates
    mc_regression_preds = mc_preds.mean(mc_dim)  #shape [batch_size, 1, sensors]
    mc_regression_error = np.abs(np.subtract(mc_regression_preds, targets))
    
    mc_regression_variance = mc_preds.std(mc_dim)
    
    mc_error = np.abs(np.subtract(mc_preds, targets[:, None, :]))
    
    mc_error_mean = mc_error.mean(mc_dim)
    mc_error_variance = mc_error.std(mc_dim) #+ mc_error_mean
    
    mc_error_entropy = compute_entropy(mc_error_mean) #
    #mc_error_entropy = compute_entropy(softmax(mc_error_mean, axis=-1))
    
    return_dict = {"orig_error": orig_error,
                   "mc_regression_error": mc_regression_error,
                   "mc_regression_variance": mc_regression_variance,
                   "mc_error_mean": mc_error_mean,
                   "mc_error_variance": mc_error_variance,
                   #"mc_error_entropy": mc_error_entropy[:, None]
                  }
                  
    return return_dict


def evaluate_uncertainty_outputs(mc_dict, gt, norm="iqr", smooth=True):
    dfs = pd.DataFrame()
    for key, val in mc_dict.items():
        val = median_iqr_norm(val,
                norm=norm, # "iqr" "mean-std",
                smooth=smooth,
                smooth_window=5)
        q = val.max(1) # np.linalg.norm(val, 2, axis=1)  #
        _, df_key = evaluate(q, gt, pa=False, interval=10, k=50, verbose=False)
        df_key = df_key.rename(columns={"without_PA": key})
        dfs = pd.concat([dfs, df_key[key]], ignore_index=False, axis=1)
    dfs = dfs.rename(index={0: "F1", 1: "Precision", 2: "Recall", 3: "AUC"})
    return dfs
    
    

def mc_pca_dim(test_array, train_array, mc_steps=10):    
    dim_1 = test_array.shape[1] -1
    step_ = int(np.ceil(dim_1 / mc_steps))
    steps = list(np.arange(1, dim_1, step_))
    steps = [dim_1] + steps
    mc_bag = []
    for dim in steps:
        pca = PCA(n_components=dim, svd_solver='full')
        pca.fit(train_array)
        t_pca = pca.transform(test_array)
        inv_pca = pca.inverse_transform(t_pca)        
        mc_bag.append(inv_pca)
    return np.stack(mc_bag, axis=1)

def mc_pca(test_array, train_array, dim=2, mc_steps=10):    
    pca = PCA(n_components=dim, svd_solver='full')
    s = len(train_array)
    step_ = int(np.ceil(s / mc_steps))
    steps = list(np.arange(0, s, step_))
    mc_bag = []
    for i in steps:
        
        pca.fit(train_array[i:i + step_:, :])
        t_pca = pca.transform(test_array)
        inv_pca = pca.inverse_transform(t_pca)        
        mc_bag.append(inv_pca)
    return np.stack(mc_bag, axis=1)