#!/usr/bin/env python3
"""Cut the panel-a schematic out of the published figure (llm_impact_v17.png, 3600 x 2946 px, 300 dpi).

NOT needed to redraw the figure; kept for provenance. Rows 0-1545 hold panel a; the three count
captions under the boxes are blanked so that make_fig5.py writes them from the data.
Run: python fig5_citations/prep/make_schematic_asset.py /path/to/llm_impact_v17.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

im = plt.imread(sys.argv[1])[:1545, :, :3].copy()
for x0, x1 in [(400, 745), (1570, 2030), (2610, 3360)]:     # count captions of the three boxes
    im[1288:1397, x0:x1] = 1.0
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "panel_a_schematic.png")
plt.imsave(out, np.clip(im, 0, 1), dpi=300)
print("->", out, im.shape)
