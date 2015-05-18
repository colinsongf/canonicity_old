from lxml import etree, objectify
import os
import csv

def genTrainFile(src_folder, tar_folder):
	from os import listdir
	from os.path import isfile, join
	files = [join(src_folder, f) for f in listdir(src_folder) if isfile(join(src_folder, f))]
	root = etree.Element("AffiliationCollection")

	for f in files:
		affiliations = parseRawFile(f)
		for aff in affiliations:
			root.insert(0, aff)
	
	f_out = open(join(tar_folder, "affiliation_data.xml"), "w")
	f_out.write(etree.tostring(root, pretty_print=True))
	f_out.close()
	# root.write(join(tar_folder, "affiliation_data.xml"), 
	# 	pretty_print=True, xml_declaration=True, encoding='UTF-8')


def parseRawFile(f):
	parser = etree.XMLParser(remove_blank_text=True)
	tree = etree.parse(f, parser)
	root = tree.getroot()

	for elem in root.getiterator():
	    if not hasattr(elem.tag, 'find'): continue  # (1)
	    i = elem.tag.find('}')
	    if i >= 0:
	        elem.tag = elem.tag[i+1:]
	objectify.deannotate(root, cleanup_namespaces=True)

	return tree.findall(".//affiliation")



# appends a labeled list to an existing xml file
# calls: appendListToXML, stripFormatting
def appendListToXMLfile(labeled_list, module, filepath):
    # format for labeled_list:      [   [ (token, label), (token, label), ...],
    #                                   [ (token, label), (token, label), ...],
    #                                   [ (token, label), (token, label), ...],
    #                                   ...           ]

    if os.path.isfile(filepath):
        with open( filepath, 'r+' ) as f:
            tree = etree.parse(filepath)
            collection_XML = tree.getroot()
            collection_XML = stripFormatting(collection_XML)

    else:
        collection_tag = module.GROUP_LABEL
        collection_XML = etree.Element(collection_tag)

    parent_tag = module.PARENT_LABEL
    collection_XML = appendListToXML(labeled_list, collection_XML, parent_tag)

    with open(filepath, 'w') as f :
        f.write(etree.tostring(collection_XML, pretty_print = True)) 


# given a list of labeled sequences to an xml list, 
# appends corresponding xml to existing xml
# calls: sequence2XML
# called by: appendListToXMLfile
def appendListToXML(list_to_append, collection_XML, parent_tag) :
    # format for list_to_append:    [   [ (token, label), (token, label), ...],
    #                                   [ (token, label), (token, label), ...],
    #                                   [ (token, label), (token, label), ...],
    #                                   ...           ]
    for labeled_sequence in list_to_append:
        sequence_xml = sequence2XML(labeled_sequence, parent_tag)
        collection_XML.append(sequence_xml)
    return collection_XML


# given a labeled sequence, generates xml for that sequence
# called by: appendListToXML
def sequence2XML(labeled_sequence, parent_tag) :
    # format for labeled_sequence:  [(token, label), (token, label), ...]

    sequence_xml = etree.Element(parent_tag)

    for token, label in labeled_sequence:
        component_xml = etree.Element(label)
        component_xml.text = token
        component_xml.tail = ' '
        sequence_xml.append(component_xml)
    sequence_xml[-1].tail = ''
    return sequence_xml


# clears formatting for an xml collection
def stripFormatting(collection) :
    collection.text = None 
    for element in collection :
        element.text = None
        element.tail = None
        
    return collection


# writes a list of strings to a file
def list2file(string_list, filepath):
    with open(filepath, 'wb') as csvfile:
        writer = csv.writer(csvfile, doublequote=True, quoting=csv.QUOTE_MINIMAL)
        for string in string_list:
            writer.writerow([string.encode('utf-8')])`