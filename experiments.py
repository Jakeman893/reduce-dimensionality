import pandas as pd 
import seaborn as sns
from classification_plots import plot_confusion_matrix, plot_learning_curve
from data_utils import enumerate_strings, normalize_data
from decomp import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Experiment:
    """"""
    def __init__(self, df, target, include, exclude, set_name = None, random_seed = None):
        self.df = df
        self.target = target
        self.include_reduced = include
        self.include = [col for col in df.columns if not col in exclude]
        self.random_seed = random_seed
        self.title = None
        self.set_name = set_name
        self.mappings = None
        self.X = None
        self.y = None

    def preprocess(self):
        self.mappings = enumerate_strings(self.df)
        self.df = normalize_data(self.df, self.target)
        self.df.head()

    def scatter_mat(self, diag_kind = 'kde'):
        sns.pairplot(self.df, hue=self.target, vars=self.include, diag_kind=diag_kind)

    def reduced_set(self):
        self.X = self.df[self.include_reduced]
        print self.X.head()
        self.y = self.df[self.target]
        self.title = 'Manual Reduce'

    def decomp_set(self, decomp_type, n_components):
        self.X = get_decomp(self.df[self.include], decomp_type, components=n_components, seed=self.random_seed)
        self.y = self.df[self.target]
        self.title = str.upper(decomp_type)

    def plt_decomp(self):
        fig = plt.figure(1, figsize=(6, 5))
        ax = Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)

        ax.scatter(self.X.T[0], self.X.T[1], self.X.T[2], c=self.y,
                edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(self.set_name + " | " + self.title)
        ax.dist = 12

    def groundTruth(self, n_components):
        # Plot the ground truth
        fig = plt.figure(1, figsize=(6, 5))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        for label in range(0,n_components):
            ax.text3D(self.X[self.y == label][self.include_reduced[0]].mean(),
                    self.X[self.y == label][self.include_reduced[1]].mean(),
                    self.X[self.y == label][self.include_reduced[2]].mean() + 2, self.mappings[self.target].inverse_transform(label),
                    horizontalalignment='center',
                    bbox=dict(alpha=.8, edgecolor='w', facecolor='w'))

        # Reorder the labels to have colors matching the cluster results
        ax.scatter(self.X[self.include_reduced[0]], self.X[self.include_reduced[1]], self.X[self.include_reduced[2]], c=self.y, edgecolor='k')
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel(self.include_reduced[0])
        ax.set_ylabel(self.include_reduced[1])
        ax.set_zlabel(self.include_reduced[2])
        ax.set_title(self.set_name + " | " + self.title + " | " + 'Ground Truth')
        ax.dist = 12
    
    def k_means_n_clust(self, n_clust):
        estimators = [('k_means', KMeans(n_clusters=n_clust, random_state=self.random_seed)),
                    ('k_means_rand_init', KMeans(n_clusters=n_clust, n_init=1,
                                                    init='random', random_state=self.random_seed))]

        fig = plt.figure(figsize=(12, 5))

        i=0
        for name, est in estimators:
            i+=1
            fig = plt.figure(i, figsize=(12, 5))
            ax = fig.add_subplot(1, 2, i, projection='3d') # Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)
            est.fit(self.X)
            labels = est.labels_

            print est.score(self.X, self.y)
            
            if type(self.X) == pd.DataFrame:
                ax.scatter(self.X[self.include_reduced[0]], self.X[self.include_reduced[1]], self.X[self.include_reduced[2]],
                        c=labels, edgecolor='k')
                ax.set_xlabel(self.include_reduced[0])
                ax.set_ylabel(self.include_reduced[1])
                ax.set_zlabel(self.include_reduced[2])
            else:
                ax.scatter(self.X.T[0], self.X.T[1], self.X.T[2],
                        c=labels, edgecolor='k')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')

            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            ax.set_title(self.set_name + " | " + self.title + " | " + name)
            ax.dist = 12

    def gauss_mix(self, n_components):
        estimators = [('gaussian_3_full', GaussianMixture(n_components=n_components, random_state=self.random_seed)),
                    ('gaussian_3_tied', GaussianMixture(n_components=n_components, covariance_type='tied',random_state=self.random_seed)),
                    ('gaussian_3_diag', GaussianMixture(n_components=n_components, covariance_type='diag', random_state=self.random_seed)),
                    ('gaussian_3_spherical', GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=self.random_seed))]

        fig = plt.figure(figsize=(12, 10))

        i=0
        for name, est in estimators:
            i+=1
            plt.subplot(2,2,i)
            ax = fig.add_subplot(2, 2, i, projection='3d') # Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)
            est.fit(self.X)
            labels = est.predict(self.X)

            print est.score(self.X, self.y)

            if type(self.X) == pd.DataFrame:
                ax.scatter(self.X[self.include_reduced[0]], self.X[self.include_reduced[1]], self.X[self.include_reduced[2]],
                        c=labels, edgecolor='k')
                ax.set_xlabel(self.include_reduced[0])
                ax.set_ylabel(self.include_reduced[1])
                ax.set_zlabel(self.include_reduced[2])
            else:
                ax.scatter(self.X.T[0], self.X.T[1], self.X.T[2],
                        c=labels, edgecolor='k')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')

            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            ax.set_title(self.set_name + " | " + self.title + " | " + name)
            ax.dist = 12

    def neural_net(self):
        estimators = [('neural_network_default', MLPClassifier()),
                    ('neural_network_prev_optimal', MLPClassifier(solver='sgd', activation='identity', hidden_layer_sizes=(13,13,13), random_state=self.random_seed))]

        fig = plt.figure(figsize=(12, 5))
        i=0
        for name, est in estimators:
            i+=1
            ax = fig.add_subplot(1, 2, i, projection='3d') # Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)
            est.fit(self.X, self.y)
            labels = est.predict(self.X)

            print est.score(self.X, self.y)

            if type(self.X) == pd.DataFrame:
                ax.scatter(self.X[self.include_reduced[0]], self.X[self.include_reduced[1]], self.X[self.include_reduced[2]],
                        c=labels, edgecolor='k')
                ax.set_xlabel(self.include_reduced[0])
                ax.set_ylabel(self.include_reduced[1])
                ax.set_zlabel(self.include_reduced[2])
            else:
                ax.scatter(self.X.T[0], self.X.T[1], self.X.T[2],
                        c=labels, edgecolor='k')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')

            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            ax.set_title(self.set_name + " | " + self.title + " | " + name)
            ax.dist = 12