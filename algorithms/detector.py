import os

from joblib import dump, load

from parameter import Configuration


class BaseDetector(Configuration):

    def __init__(self):
        Configuration.__init__(self)
        self.alg_name = 'BaseDetector'

        self.model = ''

        output_file = os.path.join(self.output_dir, 'models_dumping')
        self.model_file = os.path.join(output_file, f'{self.alg_name}.joblib')

    def train(self, X='', y=None, val_set=None):
        self.model.fit(x=X, y=y)

    def test(self, X='', y=''):
        x_test = X
        y_test = y

        self.model.predict(x=x_test)

    def dump_model(self, mode_file=''):
        if mode_file != '':
            self.model_file = mode_file
        print(f'dump mode_file: {self.model_file}')
        dump(self.model, self.model_file)

    def load_model(self, mode_file=''):
        if mode_file != '':
            self.model_file = mode_file
        print(f'load mode_file: {self.model_file}')
        self.model = load(mode_file)
