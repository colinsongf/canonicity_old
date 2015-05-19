import preprocess
import logging

def createCRFPPData(corpus_folder, output_folder, eval_folder, split_ratio):
	total_example = 0
	files = preprocess.getFiles(corpus_folder)
	logging.info(files.length, " tei files")
	for f in files:
		