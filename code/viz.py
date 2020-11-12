import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

class MplColorHelper:

  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)

def get_color_str(word, timepoint):
    if word in ['n', 'v']:
        curr_str = f'{word.upper()}'
        if timepoint in [3, 18]:
            color = 'r'
            curr_str += '1'
        elif timepoint in [6, 15]:
            color = 'g'
            curr_str += '2'
        elif timepoint in [9, 12]:
            color = 'b'
            curr_str += '3'
    else:
        curr_str = ''
        color = 'k'
    return color, curr_str


def plot_GAT(scores, noun_number, words, title):
    num_words = len(words)
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    # GAT
    im = axs[0].matshow(scores, vmin=0, vmax=1., cmap='RdBu_r', origin='lower', extent=[0, len(words), 0, len(words)])
    axs[0].xaxis.set_ticks_position('bottom')
    axs[0].set_xlabel('Testing Time (s)')
    axs[0].set_ylabel('Training Time (s)')
    axs[0].set_xticks(np.arange(0.5, len(words) + 0.5))
    axs[0].set_xticklabels(words)
    axs[0].set_yticks(np.arange(0.5, len(words) + 0.5))
    axs[0].set_yticklabels(words)
    cbar = plt.colorbar(im, ax=axs[0])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('AUC', rotation=270)
    for t in range(2, 20, 3):
        axs[0].axvline(t, color='k', ls='--')
        axs[0].axhline(t, color='k', ls='--')
    # DIAG
    axs[1].plot(range(1, num_words+1), np.diag(scores), label='diag')
    noun_position = [2, 5, 8][noun_number-1]
    axs[1].plot(range(1, num_words + 1), scores[noun_position, :], label=f'noun {noun_number}')
    axs[1].axhline(.5, color='k', linestyle='--', label='chance')
    axs[1].axvline(noun_position + 1, color='g', linestyle='--')
    axs[1].axvline(len(words) - noun_position, color='g', linestyle='--')
    axs[1].set_xlim([1, num_words+1])
    axs[1].set_ylim([-0.1, 1.1])
    axs[1].set_xlabel('Times')
    axs[1].set_ylabel('AUC')
    axs[1].set_xticks(range(1, len(words)+1))
    axs[1].set_xticklabels(words)
    axs[1].legend(loc=4)
    axs[1].axvline(.0, color='k', ls='-')

    fig.suptitle(title)

    return fig, axs


def plot_weight_trajectory(weights, mean_activation_projected, diag_scores, words, dim='2d', v_highlight=None):
    '''

    :param weights: dict with keys for grammatical number (1, 2, ..)
    :param words: list of strings. Words of sentence
    :param dim: dimensiontality of the plot
    :return: ax: matplotlib axis object
    '''
    # Number colors
    number_cmaps = {1:'Reds', 2:'Greens', 3:'Blues'}
    number_colors = {1: 'r', 2: 'g', 3: 'b'}
    st_dict = {1: 1, 2: 1, 3: 1}  # noun_number:st, counting from ONE (i.e., word position)
    ed_dict = {1: 20, 2: 20, 3: 20}  # noun_number:ed, counting from ONE (i.e., word position)

    #

    fig_pca, ax = plt.subplots(figsize=(10, 10))


    if dim == '3d':
        ax = fig_pca.gca(projection='3d')
    else:
        plt.axis('off')


    for v in range(3):


        COL = MplColorHelper(number_cmaps[v+1], 0, 1)
        st = st_dict[v + 1]
        ed = ed_dict[v + 1]
        for i_vec, (w, w_next, curr_vec, next_vec, mean_activation_sing, mean_activation_sing_next, mean_activation_plur, mean_activation_plur_next) in enumerate(
                zip(words[(st - 1):ed], words[st:(ed + 1)], weights[v+1][(st - 1):ed, :],
                    weights[v+1][st:(ed + 1), :], mean_activation_projected[v+1]['singular'][(st - 1):ed, :],
                mean_activation_projected[v + 1]['singular'][st:(ed + 1), :],mean_activation_projected[v+1]['plural'][(st - 1):ed, :],
                mean_activation_projected[v + 1]['plural'][st:(ed + 1), :])):
            color, curr_str = get_color_str(w, st + i_vec)
            c = COL.get_rgb(diag_scores[v+1][st + i_vec])
            alpha = 1
            lw = 1.5
            if v_highlight is not None:
                if v != v_highlight:
                    alpha = 0.05 # reduce alpha of trajectories not under focus
                    lw = 0.5
            if dim == '2d':
                # PLOT WEIGHT VECTORS
                x, y = curr_vec
                x_next, y_next = next_vec
                ax.quiver(x, y, x_next - x, y_next - y, scale_units='xy', angles='xy', scale=1, units='dots', width=lw,
                          color=number_colors[v + 1], alpha=alpha)
                ax.text(x, y, curr_str, fontsize=16, color=color, alpha=alpha)
                ax.scatter(x, y, c=c, alpha=alpha)
                # PLOT ACTIVATIONS
                # x_sing, y_sing = mean_activation_sing
                # x_sing_next, y_sing_next = mean_activation_sing_next
                # x_plur, y_plur = mean_activation_plur
                # x_plur_next, y_plur_next = mean_activation_plur_next
                #ax.quiver(x_sing, y_sing, x_sing_next - x_sing, y_sing_next - y_sing, scale_units='xy', angles='xy', scale=1, units='dots', width=lw,
                #          color='r', alpha=alpha)
                #ax.quiver(x_plur, y_plur, x_plur_next - x_plur, y_plur_next - y_plur, scale_units='xy', angles='xy',
                #          scale=1, units='dots', width=lw, linestyle='--',
                #          color='b', alpha=alpha)

            elif dim == '3d':
                x, y, z = curr_vec
                x_next, y_next, z_next = next_vec
                ax.quiver(x, y, z, x_next - x, y_next - y, z_next - z, facecolors=number_colors[v+1], colors=number_colors[v+1], arrow_length_ratio=0.1, alpha=alpha)
                ax.text(x, y, z, curr_str, fontsize=16, color=color, alpha=alpha)
                ax.scatter(x, y, z, c=c, alpha=alpha)


            # Add last point, at the end of the last arrow
            color, curr_str = get_color_str(w_next, st + i_vec + 1)
            if dim == '2d':
                ax.text(x_next, y_next, curr_str, fontsize=16, color=color, alpha=alpha)
                ax.scatter(x_next, y_next, color=COL.get_rgb(diag_scores[v+1][st + i_vec]), alpha=alpha)
            elif dim == '3d':
                ax.text(x_next, y_next, z_next, curr_str, fontsize=16, color=color, alpha=alpha)
                ax.scatter(x_next, y_next, z_next, color=COL.get_rgb(diag_scores[v+1][st + i_vec]), alpha=alpha)
    return fig_pca, ax