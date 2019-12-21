import os
import sys
import subprocess
import shutil
import concurrent.futures
from pathlib import Path
import xml.etree.ElementTree as ET

# Converts objs used by a scene file into a serialized representation
# Greatly improves scene loading time

MTSIMPORT = shutil.which('mtsimport') # must be in PATH
if MTSIMPORT is None:
	raise Exception('Executable mtsimport not in PATH, cannot proceed')

if "-h" in sys.argv or "--help" in sys.argv or len(sys.argv) < 2:
	print("Usage: python3 {} scenefile.xml [scenefile2.xml ...]".format(os.path.basename(__file__)))
	sys.exit(0)

def serialize(file):
	dummy_xml = file.with_suffix(".xml") # generates .serialized with same name
	
	proc_cmds = [
		MTSIMPORT,
		str(file.resolve()),
		str(dummy_xml.resolve())
	]
		
	current_subprocess = subprocess.Popen(proc_cmds, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	current_subprocess.wait()
	
	os.remove(dummy_xml)

INFILES = sys.argv[1:]

for INFILE in INFILES:
	OUTFILE = INFILE.replace(".xml", "_serialized.xml")
	print("\nSerializing: {} => {}".format(INFILE, OUTFILE))

	# Parse scenefile
	root = Path(INFILE).parent
	scene = ET.parse(INFILE)
	
	# Find all unconverted meshes
	objs_to_do = set()
	for shape in scene.iter("shape"):
		for k in shape.iter("string"):
			if k.attrib["name"] == "filename":
				target = (root / k.attrib["value"]).with_suffix(".serialized")
				if not target.is_file():
					objs_to_do.add(target.with_suffix(".obj"))
					
	# Convert objects in parallel
	with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
		results = [executor.submit(serialize, obj) for obj in objs_to_do]

		total = len(results)
		for i, res in enumerate(results):
			res.result()
			print("\rProcessing mesh {}/{}".format(i + 1, total), end="")
	
	# Create updated xml
	for shape in scene.iter("shape"):
		shape.attrib["type"] = "serialized"
		for k in shape.iter("string"):
			if k.attrib["name"] == "filename":
				k.attrib["value"] = k.attrib["value"].replace(".obj", ".serialized")
		
	# Output new scenefile
	scene.write(OUTFILE)