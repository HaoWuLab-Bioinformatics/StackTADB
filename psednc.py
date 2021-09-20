from repDNA.psenac import PseDNC
import numpy as np

psednc = PseDNC(lamada=1, w=0.05)

x = psednc.make_psednc_vec(open('data_base.fasta'))
np.savetxt('feature_psednc.txt',x)