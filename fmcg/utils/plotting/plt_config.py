import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
plt.style.use('default')

# point to local font file
# font_path = "/data/fonts/helvetica/Helvetica.ttf"
# prop = font_manager.FontProperties(fname=font_path)
font_dir = ['/data/fonts']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

# Use Helvetica only if the font is actually installed; otherwise keep matplotlib's
# default sans-serif. Setting it unconditionally makes every plot emit a "font not
# found" warning and silently fall back on machines without the Helvetica .ttf.
if 'Helvetica' in {f.name for f in font_manager.fontManager.ttflist}:
    plt.rcParams['font.family'] = 'Helvetica'
mpl.rcParams["font.size"] = 8


mpl.rcParams['figure.figsize'] = (7.11, 3)
mpl.rcParams['figure.dpi'] = 500
mpl.rcParams['mathtext.fontset'] = 'stix'
#mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['axes.titlepad'] = 5 
mpl.rcParams['xtick.direction'] = "in"
mpl.rcParams['ytick.direction'] = "in"
# mpl.rcParams['xtick.major.pad'] = 2
# mpl.rcParams['ytick.major.pad'] = 2
# mpl.rcParams['xtick.minor.pad'] = 2
# mpl.rcParams['ytick.minor.pad'] = 2
# mpl.rcParams['ytick.labelsize'] = "medium"
# mpl.rcParams['xtick.labelsize'] = "medium"
mpl.rcParams['axes.axisbelow'] = "line"
mpl.rcParams['axes.xmargin'] = 0.0
mpl.rcParams['axes.ymargin'] = 0.01
mpl.rcParams['axes.zmargin'] = 0.01
# plt.rcParams['axes.labelsize'] = 8  # or even 6-7 depending on figure size
# plt.rcParams['xtick.labelsize'] = 6
# plt.rcParams['ytick.labelsize'] = 6


