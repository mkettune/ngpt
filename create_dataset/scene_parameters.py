#########################################
# Sampling camera positions from scenes #
#########################################

import numpy as np
import random
from math import sin,cos,acos,exp

#Array values will be comma delimited
def TupleOfArraysToTupleOfStrings( tup ):
	res = ()
	for e in tup:
		str = ','.join(np.array(e).astype(np.str))
		res = res + (str,)
	return res

def SampleUnitSphere():
	theta = 2.0 * np.pi * random.uniform(0,1)
	phi = acos(1.0 - 2.0 * random.uniform(0,1))
	return np.array([sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi)])

def PointInsideBB( bbmin, bbmax, point ):
	if (bbmin[0] <= point[0] <= bbmax[0] and
		bbmin[1] <= point[1] <= bbmax[1] and
		bbmin[2] <= point[2] <= bbmax[2]):
		return True
	else:
		return False
		
#return value: ( beginTarget, endTarget, beginOrigin, endOrigin, beginUp, endUp )
#as a tuple of comma delimited strings of coordinates i.e. ( "1.0, 2.0, 3.0", ... )
def SampleSensorParameters( sceneName, seed = None, useTargets = True, useDof = True ):
	# Scene configurations: Occlusion-free bounding box for the camera and a velocity scale for motion blur.
	
	# scene: ( CamBB_Scale, CamBB_Origin, Cam_velocity )
	# cam_vel / bb_diag_len: min = 62, max = 575, mean = 190, median = 102
	dictCamBB = { 
		'bathroom': 				( np.array([ 3.5, 2.5, 3.5 ]), np.array([ -6.0, 5.5, 6.0 ]), 0.075 ),
		'bookshelf_rough2': 		( np.array([ 5.0, 3.0, 5.0 ]), np.array([ 10.0, 5.0, 18.0 ]), 0.050645 ),
		'crytek_sponza': 			( np.array([ 850.0, 500.0, 150.0 ]), np.array([ -75.0, 600.0, -40.0 ]), 7.506946*0.5 ),
		'hairball': 				( np.array([ 15.0, 15.0, 15.0 ]), np.array([ 0.0, 0.0, 0.0 ]), 0.420264 ),
		'jogging': 					( np.array([ 3.25, 1.4, 3.25 ]), np.array([ 0.0, 1.7, 0.0 ]), 0.048464 ),
		'kitchen_simple': 			( np.array([ 150.0, 66.0, 150.0 ]), np.array([ -75.0, 175.0, 300.0 ]), 0.386664 ),
		'new_bedroom': 				( np.array([ 1.5, 0.66, 1.25 ]), np.array([ 0.5, 1.25, 1.5 ]), 0 ),
		'new_dining_room': 			( np.array([ 4.0, 2.0, 3.0 ]), np.array([ -0.5, 4.0, -0.5 ]), 0 ),
		'new_kitchen_animation':	( np.array([ 1.25, 0.75, 2.0 ]), np.array([ 0.0, 2.0, 0.0 ]), 0.024227 ),
		'bathroom2': 				( np.array([ 0.8, 0.5, 0.6 ]), np.array([ -1, 1.5, -1.4 ]), 'auto' ),
		'bathroom3': 				( np.array([ 8, 7, 20 ]), np.array([ 3, 15, 5 ]), 'auto' ),
		'classroom': 				( np.array([ 2, 0.3, 3 ]), np.array([ 0, 1.6, 0 ]), 'auto' ),
		'living-room': 				( np.array([ 1.3, 0.5, 1.2 ]), np.array([ 2, 1.5, -1.7 ]), 'auto' ),
		'living-room-2': 			( np.array([ 1.5, 0.4, 1.3 ]), np.array([ 0.3, 1.5, 2.7 ]), 'auto' ),
		'living-room-3': 			( np.array([ 1.8, 0.35, 1.1 ]), np.array([ -2.7, 1, -2.5 ]), 'auto' ),
		'staircase': 				( np.array([ 1.3, 2.2, 0.7 ]), np.array([ 0.1, 2.8, -0.5 ]), 'auto' ),
		'staircase2': 				( np.array([ 5, 2.0, 0.8 ]), np.array([ -0.8, 2.4, 1.5 ]), 'auto' ),
	}
	
	# Calculate speed if none given
	for key, (scale, orig, velocity) in dictCamBB.items():
		if velocity == 'auto':
			diag = np.linalg.norm(scale)
			dictCamBB[key] = (scale, orig, diag / 100)

	
	# Interesting view locations for the scenes.
	
	#scene: [ target_0, target_1, ..., target_n ]
	dictTargets = {
		# right window, left window,
		# bathtub reflection, wall reflection
		'bathroom': 				[ np.array([0.614863, 4.95646, 2.41749]), np.array([-2.43959, 4.72377,-1.27463]),
									  np.array([-10.5439, 1.33694, 2.72729]), np.array([-11.8331, 5.39295, 3.02594]) ],
		# teapot, above the lamp,
		# in front of the cabinet windows, painting
		'bookshelf_rough2': 		[ np.array([ 9.2728,  3.8543,  11.1554]), np.array([ 15.0546, 9.58031, 11.5428]), 
									  np.array([15.3744, 6.61799,  14.3623]), np.array([ 17.0251, 5.82917, 18.3563]) ],
		'crytek_sponza': 			[ ],
		'hairball': 				[ ],
		# guy
		'jogging': 					[ np.array([0.0, 1.40, 0.10])],
		# bright floor1, bright floor2
		# kitchen counter down low
		'kitchen_simple': 			[ np.array([-311.768, 13.6271, 478.134]), np.array([-163.503, 11.3999, 499.62 ]),
									  np.array([-34.5865, 37.7344, 274.899]) ],
		# window low, window mid
		# faucet1, faucet2,
		# shelves
		'new_bathroom': 			[ np.array([-2.53223, 0.57276,-1.37019]), np.array([-2.50905, 1.51837,-1.42284]),
									  np.array([ -1.8441, 1.01598,-2.24424]), np.array([-0.85426, 1.00074,-2.25948]), 
									  np.array([-2.11828, 1.71587,-0.22450]) ],
		# curtain rod1, curtain rod2
		# left curtain edge, right curtain edge
		'new_bedroom': 				[ np.array([ 0.79871, 2.32616,-0.99425]), np.array([-0.88004, 2.33150,-0.97539]),
									  np.array([-2.12700, 0.94562,-0.97755]), np.array([0.952772, 1.11918,-1.12454]) ],
		# between teapot and jug, cup1
		# cup2, cup3
		# lamp1
		'new_dining_room': 			[ np.array([-0.43342, 1.22514,-2.16258]), np.array([ 1.51204, 1.08712,-0.98936]),
									  np.array([-1.03737, 1.01173,-1.13951]), np.array([-2.94684, 1.05917,-1.90627]),
									  np.array([ 1.48275, 3.86904,-2.00124]) ],
		# window, utensils
		# pots, dining table
		'new_kitchen_animation':	[ np.array([0.082465, 1.76982,-2.69474]), np.array([-2.46713, 1.26516,-1.03350]),
									  np.array([-2.06267, 1.20503, 0.21826]), np.array([ 0.21825, 1.13351,-1.31494]) ],
		# lightbulb 254, faucet 589,
		# ball 604, picture 205,
		# mirror 602
		'bathroom2': 				[ np.array([-1.23, 1.97, -2.20]), np.array([-1.74, 1.00, -2.37]),
									  np.array([-1.40, 0.85, -2.29]), np.array([-1.94, 1.73, -0.22]),
									  np.array([-1.40, 1.49, -2.43]) ],
		# trashcan 054, figure 019,
		# round mirror 017
		'bathroom3': 				[ np.array([-5.98, 2.69, 0.40]), np.array([-6.86, 11.18, -22.36]),
									  np.array([-8.18, 9.87, 20.06]) ],
		# flag 005, desks 064,
		# front desk 017
		'classroom': 				[ np.array([-3.70, 1.20, -4.94]), np.array([-1.14, 0.92, -1.03]),
									  np.array([-1.94, 1.25, -4.70]) ],
		# bottle 002, horse 012,
		# painting 023, plant 058,
		# table leg 037
		'living-room': 				[ np.array([2.53, 0.77, -1.87]), np.array([1.85, 1.01, -0.10]),
									  np.array([1.90, 1.55, -3.61]), np.array([0.74, 0.49, 0.07]),
									  np.array([2.58, 0.21, -1.89]) ],
		# green pillow 160, lampshade 171,
		# fruit bowl 088, teapot 002,
		# painting 169, fireplace 024,
		# books 109, radio 163,
		# onyx bowls 087
		'living-room-2': 			[ np.array([0.56, 0.66, 1.59]), np.array([2.78, 1.59, 1.84]),
									  np.array([0.25, 0.46, 2.94]), np.array([0.97, 0.60, 2.96]),
									  np.array([2.98, 1.75, 2.99]), np.array([-1.84, 0.27, 2.88]),
									  np.array([-2.25, 0.99, 3.86]), np.array([-2.22, 1.30, 1.52]),
									  np.array([-1.78, 1.33, 3.13]) ],
		
		# painting 032, branches 034,
		# table 023, glass vase 021,
		# glass spheres 015, carpet edge
		'living-room-3': 			[ np.array([-4.01, 1.50, -0.04]), np.array([-3.11, 1.18, -0.31]),
									  np.array([-1.42, 0.22, -1.91]), np.array([-1.48, 1.66, -0.11]),
									  np.array([-3.16, 0.53, -0.81]), np.array([-2.62, 0.00, -1.95]) ],
		# chair 577, lamp 742,
		# knob 408, chandelier 049,
		# picture_glasses 704,
		# stairs 313
		'staircase': 				[ np.array([-0.12, 0.72, -2.54]), np.array([-0.75, 1.04, -3.92]),
									  np.array([0.53, 2.04, -1.46]), np.array([-0.53, 4.31, -2.03]),
									  np.array([1.59, 4.48, -5.13]), np.array([0.53, 2.49, -3.21]) ],
		# glass pane middle 014, glass pane left edge 014,
		# wallpaper stripe 018, stairs 001,
		# stairs lower 001
		'staircase2': 				[ np.array([-0.74, 0.73, -0.04]), np.array([1.91, 0.87, -0.01]),
									  np.array([-7.98, 1.90, -3.00]), np.array([-0.85, 2.73, -1.56]),
									  np.array([-2.86, -0.43, -2.05])],
	}
	
	scale = dictCamBB[sceneName][0]
	origin = dictCamBB[sceneName][1]
	v = dictCamBB[sceneName][2]
	
	bbmin = origin-scale
	bbmax = origin+scale
	
	if seed is not None:
		random.seed(seed)
	
	targetDir = SampleUnitSphere()
	
	beginOrigin = np.array([random.uniform(bbmin[0],bbmax[0]),random.uniform(bbmin[1],bbmax[1]),random.uniform(bbmin[2],bbmax[2])])
	beginTarget = beginOrigin + targetDir
	
	sigma = v / 1.596
	
	vgauss = np.array([ random.gauss(0, sigma), random.gauss(0, sigma), random.gauss(0, sigma) ])
	endOrigin = beginOrigin + vgauss
	while( not PointInsideBB( bbmin, bbmax, endOrigin ) ):
		vgauss = np.array([ random.gauss(0, sigma), random.gauss(0, sigma), random.gauss(0, sigma) ])
		endOrigin = beginOrigin + vgauss

	endTarget = endOrigin + targetDir

	focusDist = 1.0
	apertureRadius = 0.0
	
	if useTargets:
		n = len(dictTargets[sceneName])
		if n > 0:
			i = random.randint(0,n-1)
			beginTarget = dictTargets[sceneName][i]
			endTarget = beginTarget
			
			if useDof:
				focusDist = np.linalg.norm(beginTarget - beginOrigin)
				apertureRadius = abs(random.gauss(0, 0.005 * focusDist))
				
	
	beginUp = SampleUnitSphere()
	endUp = beginUp
	
	return TupleOfArraysToTupleOfStrings( (beginTarget,endTarget,beginOrigin,endOrigin,beginUp,endUp,[focusDist],[apertureRadius]) )
	