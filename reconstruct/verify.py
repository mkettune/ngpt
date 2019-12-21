import subprocess
import os
import numpy as np

os.environ['DARC_ROOT'] = r'F:\neuralgpt_drive'

subprocess.run(r'python F:\neuralgpt_drive\cs\graphics\kettunm2\neuralgpt\code\run_results.py --config=109 --crops=4 --reconstruct')

new_image = np.load(os.environ['DARC_ROOT'] + r'\cs\graphics\kettunm2\neuralgpt\results\new_rebu\postreb__109__ind10__grad_dna__final_1__loss_elpips_squeeze_maxpool_plus_reg_pg__3evals_lhs__exp_50k_70k_0p5__avg_nearest_\viz10-0.0-bookshelf_rough2-2-12\img0000_crop04.npz')['arr_0']
ref_image = np.load(os.environ['DARC_ROOT'] + r'\cs\graphics\kettunm2\neuralgpt\results\new_rebu_final\postreb__109__ind10__grad_dna__final_1__loss_elpips_squeeze_maxpool_plus_reg_pg__3evals_lhs__exp_50k_70k_0p5__avg_nearest_\viz10-0.0-bookshelf_rough2-2-12\img0000_crop04.npz')['arr_0']

max_error = np.max(np.abs(new_image - ref_image))
if max_error > 2e-5:
	print("**** Maximum error is {}! ****".format(max_error))
else:
		print("**** Success (error {}). ****".format(max_error))
