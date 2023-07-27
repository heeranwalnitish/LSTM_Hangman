import tensorflow as tf
from tensorflow import keras
import os


class generic_model(keras.Model):
    """
    Contains basic functions for storing and loading a model
    """

    def __init__(self, config):
        super(generic_model, self).__init__()

        self.config_file = config

    def loss(self, predicted, truth):
        return self.loss_func(predicted, truth)

    # Save model, along with loss details and testing accuracy
    # The best model has the lowest test loss and is used during feature extraction
    def save_model(self, is_best, epoch, train_loss, test_loss, rnn_name, layers, hidden_dim):
        base_path = self.config_file['models']
        if is_best:
            filename = base_path + 'best_' + '_'.join([rnn_name, str(layers), str(hidden_dim)]) + '.h5'
        else:
            filename = base_path + str(epoch) + '_' + '_'.join([rnn_name, str(layers), str(hidden_dim)]) + '.h5'

        self.save_weights(filename)

        print("Saved model")

    # Loads saved model for resuming training or inference
    def load_model(self, mode, rnn_name, layers, hidden_dim, epoch=None):
        # If epoch is given, load that particular model; otherwise, load the model with the name 'best'
        if mode == 'test' or mode == 'test_one':
            try:
                if epoch is None:
                    filename = self.config_file['models'] + 'best_' + '_'.join(
                        [rnn_name, str(layers), str(hidden_dim)]) + '.h5'
                else:
                    filename = self.config_file['models'] + str(epoch) + '_'.join(
                        [rnn_name, str(layers), str(hidden_dim)]) + '.h5'

                self.load_weights(filename)

                print("Loaded pretrained model from:", filename)
            except:
                print("Couldn't find model for testing")
                exit(0)
        # Train
        else:
            # If epoch is given, load that particular model; otherwise, load the model trained on the most number of epochs
            # e.g., if the directory has 400, 500, 600, it will load 600.h5
            if epoch is not None:
                filename = self.config_file['models'] + str(epoch) + '_' + '_'.join(
                    [rnn_name, str(layers), str(hidden_dim)]) + '.h5'
            else:
                directory = [x.split('_') for x in os.listdir(self.config_file['models'])]
                to_check = []
                for poss in directory:
                    try:
                        to_check.append(int(poss[0]))
                    except:
                        continue

                if len(to_check) == 0:
                    print("No pretrained model found")
                    return 0, [], []
                # Model trained on the most epochs
                filename = self.config_file['models'] + str(max(to_check)) + '_' + '_'.join(
                    [rnn_name, str(layers), str(hidden_dim)]) + '.h5'

            self.load_weights(filename)

            print("Loaded pretrained model from:", filename)
            
        return filename

            
