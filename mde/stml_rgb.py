from PIL import ImageFont, ImageDraw, Image
import numpy as np
from tqdm import tqdm
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectKBest
import colorsys


class STML_RGB(TransformerMixin):
    def __init__(self, size=(224, 224), verbose=False, n_cols=None):
        self.size_ = size
        self.verbose_ = verbose
        self.n_cols_ = n_cols
        
    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        self.n_features_ = self.X_.shape[1]
        self.anova_ = SelectKBest(k='all').fit(self.X_, self.y_).scores_
        self.mmanova_ = self.map_value(self.anova_, np.min(self.anova_), np.max(self.anova_), 0, 1)
        self.colors_ = [list(self.get_color(self.mmanova_[i])) for i in range(len(self.mmanova_))]
        
        return self
    
    def get_color(self, red_to_green):
        assert 0 <= red_to_green <= 1
        # in HSV, red is 0 deg and green is 120 deg (out of 360);
        # divide red_to_green with 3 to map [0, 1] to [0, 1./3.]
        hue = red_to_green / 3.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        return map(lambda x: int(255 * x), (r, g, b))

    def map_value(self, value, min_value, max_value, min_result, max_result):
        result = min_result + (value - min_value)/(max_value - min_value)*(max_result - min_result)
        return result

    def transform(self, X):
        if self.n_cols_ == None:
            n_columns = 2 if self.n_features_ < 30 else 3
        else:
            n_columns = self.n_cols_

        n_rows = np.ceil(self.n_features_/n_columns)
        xs = np.ceil(self.size_[0]/n_columns)
        ys = np.ceil(self.size_[0]/n_rows)
        X = np.round(X, 4)
        # Coords for each text
        coords = []
        for i in range(int(n_rows)):
            for j in range(n_columns):
                coords.append((xs*j, ys*i))

        # Optmize font size
        X_string = list(map(str, X[:,:self.n_features_].flatten().tolist()))
        longest_feature = max(X_string, key=len)

        max_font_size = 100
        for tmp_font_size in range(1, max_font_size):
            font = ImageFont.truetype("mde/FreeSans.ttf", tmp_font_size)
            # feature_size = font.getsize(longest_feature)
            feature_size = font.getbbox(longest_feature)[2:4]
            # Stop if font is too large
            if feature_size[0] >= xs or feature_size[1] >= ys:
                break
            # A little smaller font size to ensure readability
            font_size = tmp_font_size-2

        font = ImageFont.truetype("mde/FreeSans.ttf", font_size)

        # Find longest string at a given font size for centering
        max_string_length = 100
        center_width = len(longest_feature)
        for string_length in range(len(longest_feature), max_string_length):
            tmp_longest = font.getbbox(''.join(" " for i in range(string_length+1)))[2:4]
            if tmp_longest[0] >= xs:
                break
            center_width = string_length


        # Container for full set
        data = np.zeros((X.shape[0], self.size_[0], self.size_[1], 3))
        for sample in tqdm(range(X.shape[0]), disable=not self.verbose_):
            img = np.zeros((self.size_[0], self.size_[1])).astype(np.uint8)

            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)

            rgb_img = []
            for channel in range(3):
                for id in range(self.n_features_):
                    c = self.colors_[id]
                    draw.text((coords[id][0], coords[id][1]),
                            str(X[sample,id]),
                            font=font, fill=c[channel])
                rgb_img.append(np.array(img))
            rgb_img = np.stack(rgb_img, axis=0)
            data[sample] = np.moveaxis(rgb_img, 0, 2)
        return data.astype(np.uint8)