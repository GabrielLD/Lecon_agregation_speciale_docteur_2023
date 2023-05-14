""" Diverses fonctions récurentes dans les programmes des leçons"""

import os
import matplotlib.pyplot as plt
from .utils import justify
from .widgets import make_param_widgets, make_choose_plot, make_reset_button, make_log_button, make_start_stop_animation
from .widgets import FloatSlider, IntSlider


path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

plt.style.use(os.path.join(dir_path, 'lecons.mplstyle'))



